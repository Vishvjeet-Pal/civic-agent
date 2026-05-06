import asyncio
from logging.config import fileConfig
from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config 
from app.core.config import get_settings
from app.db.session import Base
from app.db import models

config = context.config
settings = get_settings()

def get_url():
    url = settings.database_url
    if "postgres" in url:
        import socket
        try:
            socket.gethostbyname("postgres")
        except socket.gaierror:
            # Running locally outside Docker, use host port mapping
            url = url.replace("@postgres:5432", "@localhost:5433")
    return url

def get_url_sync():
    url = settings.database_url_sync
    if "postgres" in url:
        import socket
        try:
            socket.gethostbyname("postgres")
        except socket.gaierror:
            url = url.replace("@postgres:5432", "@localhost:5433")
    return url

config.set_main_option("sqlalchemy.url", get_url_sync())

if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    context.configure(url=get_url_sync(), target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    connectable = async_engine_from_config(
        {"sqlalchemy.url": get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()