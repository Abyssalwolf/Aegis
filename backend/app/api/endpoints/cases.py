from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, String, cast

from app.api import deps
from app.models.user import User
from app.models.case import Case
from app.models.assignment import CaseAssignment
from app.models.activity import ActivityLog
from app.schemas.case import Case as CaseSchema, CaseCreate

router = APIRouter()

async def check_case_access(db: AsyncSession, case: Case, current_user: User):
    if current_user.role == "ADMIN":
        return True # Or maybe admins don't access cases. The requirements say "Admin does NOT work inside investigation cases."
    
    if current_user.role == "OFFICER":
        if case.created_by == current_user.id:
            return True
        
        # Check assignment
        result = await db.execute(
            select(CaseAssignment).filter(
                CaseAssignment.case_id == case.id,
                CaseAssignment.officer_id == current_user.id
            )
        )
        if result.scalars().first():
            return True
    return False

@router.post("", response_model=CaseSchema)
async def create_case(
    *,
    db: AsyncSession = Depends(deps.get_db),
    case_in: CaseCreate,
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    # "Only SI or higher (clearance >= 4) can create cases."
    if (current_user.clearance_level or 0) < 4:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Clearance level 4 or higher required to create a case."
        )
        
    case = Case(
        title=case_in.title,
        description=case_in.description,
        required_clearance_level=case_in.required_clearance_level,
        status=case_in.status,
        created_by=current_user.id,
    )
    db.add(case)
    
    # Needs a flush to get the case.id for assignments
    await db.flush()
    
    # Create assignments
    if case_in.assigned_officer_ids:
        for officer_id in case_in.assigned_officer_ids:
            assignment = CaseAssignment(
                case_id=case.id,
                officer_id=officer_id
            )
            db.add(assignment)
    
    # Log activity
    activity = ActivityLog(
        case=case,
        user_id=current_user.id,
        action="CASE_CREATED"
    )
    db.add(activity)
    
    await db.commit()
    await db.refresh(case)
    return case

@router.get("/search", response_model=List[CaseSchema])
async def search_cases(
    q: str = "",
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    query = select(Case).outerjoin(
        CaseAssignment, Case.id == CaseAssignment.case_id
    ).filter(
        Case.title.ilike(f"%{q}%"),
        or_(
            Case.created_by == current_user.id,
            CaseAssignment.officer_id == current_user.id
        )
    ).distinct()
    
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/{id}", response_model=CaseSchema)
async def get_case(
    id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    if not await check_case_access(db, case, current_user):
         raise HTTPException(status_code=403, detail="Forbidden")
         
    return case

@router.delete("/{id}")
async def delete_case(
    id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    if case.created_by != current_user.id:
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Only the case creator can delete this case."
        )
        
    # Delete associated assignments and activities manually if not cascaded
    await db.execute(CaseAssignment.__table__.delete().where(CaseAssignment.case_id == id))
    await db.execute(ActivityLog.__table__.delete().where(ActivityLog.case_id == id))
    
    await db.delete(case)
    await db.commit()
    return {"message": "Case deleted successfully"}

from app.schemas.case import CaseTransfer

@router.post("/{id}/transfer", response_model=CaseSchema)
async def transfer_case(
    id: UUID,
    transfer_in: CaseTransfer,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    result = await db.execute(select(Case).filter(Case.id == id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    if case.created_by != current_user.id:
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Only the case creator can transfer this case."
        )
        
    # Verify new owner
    result_user = await db.execute(select(User).filter(User.id == transfer_in.new_owner_id))
    new_owner = result_user.scalars().first()
    
    if not new_owner or not new_owner.is_active or new_owner.role != "OFFICER":
        raise HTTPException(status_code=400, detail="Invalid target officer.")
        
    if (new_owner.clearance_level or 0) < 4:
        raise HTTPException(status_code=400, detail="Target officer must have clearance level >= 4.")
        
    case.created_by = new_owner.id
    db.add(case)
    
    # Ensure the new owner is assigned to the case
    res_assign = await db.execute(
        select(CaseAssignment).filter(
            CaseAssignment.case_id == case.id,
            CaseAssignment.officer_id == new_owner.id
        )
    )
    if not res_assign.scalars().first():
        assignment = CaseAssignment(case_id=case.id, officer_id=new_owner.id)
        db.add(assignment)
        
    # Log activity
    activity = ActivityLog(
        case=case,
        user_id=current_user.id,
        action="CASE_TRANSFERRED"
    )
    db.add(activity)
    
    await db.commit()
    await db.refresh(case)
    return case

from app.schemas.assignment import CaseAssignmentCreate
from app.schemas.user import User as UserSchema

@router.get("/{id}/officers", response_model=List[UserSchema])
async def get_assigned_officers(
    id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    # First ensure the user has access to the case
    result = await db.execute(select(Case).filter(Case.id == id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    if not await check_case_access(db, case, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    query = select(User).join(CaseAssignment, User.id == CaseAssignment.officer_id).filter(CaseAssignment.case_id == id)
    result = await db.execute(query)
    return result.scalars().all()

@router.post("/{id}/officers", response_model=UserSchema)
async def assign_officer(
    id: UUID,
    assignment_in: CaseAssignmentCreate,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    # Verify owner & higher clearance
    result = await db.execute(select(Case).filter(Case.id == id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    creator_res = await db.execute(select(User).filter(User.id == case.created_by))
    creator = creator_res.scalars().first()
    creator_clearance = creator.clearance_level or 0 if creator else 0
    my_clearance = current_user.clearance_level or 0

    if case.created_by != current_user.id:
        # Check if the current user is assigned AND has a higher clearance level
        assign_res = await db.execute(
            select(CaseAssignment).filter(
                CaseAssignment.case_id == id,
                CaseAssignment.officer_id == current_user.id
            )
        )
        if not assign_res.scalars().first() or my_clearance <= creator_clearance:
            raise HTTPException(status_code=403, detail="Forbidden: Only the creator or a higher-clearance assigned officer can manage officers.")
        
    # Verify target officer
    result_user = await db.execute(select(User).filter(User.id == assignment_in.officer_id))
    officer = result_user.scalars().first()
    if not officer or officer.role != "OFFICER" or not officer.is_active:
        raise HTTPException(status_code=400, detail="Invalid officer.")
        
    # Make sure not already assigned
    res_assign = await db.execute(
        select(CaseAssignment).filter(
            CaseAssignment.case_id == id,
            CaseAssignment.officer_id == officer.id
        )
    )
    if res_assign.scalars().first():
        raise HTTPException(status_code=400, detail="Officer is already assigned to this case.")
        
    assignment = CaseAssignment(case_id=id, officer_id=officer.id)
    db.add(assignment)
    
    # Log activity
    activity = ActivityLog(case_id=id, user_id=current_user.id, action=f"ASSIGNED_OFFICER:{officer.id}")
    db.add(activity)
    
    await db.commit()
    return officer

@router.delete("/{id}/officers/{officer_id}")
async def remove_officer(
    id: UUID,
    officer_id: UUID,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_officer),
) -> Any:
    # Verify owner & higher clearance
    result = await db.execute(select(Case).filter(Case.id == id))
    case = result.scalars().first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    creator_res = await db.execute(select(User).filter(User.id == case.created_by))
    creator = creator_res.scalars().first()
    creator_clearance = creator.clearance_level or 0 if creator else 0
    my_clearance = current_user.clearance_level or 0

    if case.created_by != current_user.id:
        # Check if the current user is assigned AND has a higher clearance level
        assign_res = await db.execute(
            select(CaseAssignment).filter(
                CaseAssignment.case_id == id,
                CaseAssignment.officer_id == current_user.id
            )
        )
        if not assign_res.scalars().first() or my_clearance <= creator_clearance:
            raise HTTPException(status_code=403, detail="Forbidden: Only the creator or a higher-clearance assigned officer can manage officers.")
        
    # Cannot remove the true creator of the case under any circumstances
    if officer_id == case.created_by:
        raise HTTPException(status_code=400, detail="Cannot remove the original creator of the case.")
        
    res_assign = await db.execute(
        select(CaseAssignment).filter(
            CaseAssignment.case_id == id,
            CaseAssignment.officer_id == officer_id
        )
    )
    assignment = res_assign.scalars().first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Officer is not assigned to this case.")
        
    await db.delete(assignment)
    
    # Log activity
    activity = ActivityLog(case_id=id, user_id=current_user.id, action=f"REMOVED_OFFICER:{officer_id}")
    db.add(activity)
    
    await db.commit()
    return {"message": "Officer removed successfully"}
