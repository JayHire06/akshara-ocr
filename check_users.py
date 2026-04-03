import asyncio
from api.db.database import AsyncSessionLocal
from api.db.models import User

async def check():
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(select(User))
        users = result.scalars().all()
        print(f'Total users: {len(users)}')
        for u in users:
            print(f'  Username: {u.username}')

if __name__ == "__main__":
    asyncio.run(check())
