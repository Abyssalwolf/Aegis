import asyncio
from app.db.database import AsyncSessionLocal
from app.models.user import User
from app.core.security import get_password_hash
from sqlalchemy.future import select

async def seed_admin():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).filter(User.username == "admin"))
        admin_user = result.scalars().first()
        if not admin_user:
            print("Creating default admin user...")
            admin_user = User(
                username="admin",
                hashed_password=get_password_hash("admin123"),
                role="ADMIN",
                is_active=True,
            )
            db.add(admin_user)
            await db.commit()
            print("Admin user created successfully. Username: admin, Password: admin123")
        else:
            print("Admin user already exists.")

if __name__ == "__main__":
    asyncio.run(seed_admin())
