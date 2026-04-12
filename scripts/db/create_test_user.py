import asyncio
from api.db.database import AsyncSessionLocal
from api.db.models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def get_password_hash(password):
    return pwd_context.hash(password)

async def main():
    async with AsyncSessionLocal() as session:
        user = User(username="testuser", hashed_password=get_password_hash("testpass"))
        session.add(user)
        try:
            await session.commit()
            print("Test user created successfully.")
        except Exception as e:
            print("User probably exists or error:", e)

if __name__ == "__main__":
    asyncio.run(main())
