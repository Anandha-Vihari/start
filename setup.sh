#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting setup..."

# Update package lists
echo "🔄 Updating package lists..."
sudo apt update -y

# Install PostgreSQL if not installed
if ! command -v psql &> /dev/null; then
    echo "📦 Installing PostgreSQL..."
    sudo apt install -y postgresql postgresql-contrib
else
    echo "✅ PostgreSQL is already installed."
fi

# Start PostgreSQL service
echo "🚀 Starting PostgreSQL service..."
sudo service postgresql start

# Switch to postgres user and setup database
echo "📂 Setting up the database..."
sudo -u postgres psql <<EOF
CREATE DATABASE your_database;
CREATE USER username WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE your_database TO username;
ALTER USER username CREATEDB;
EOF

echo "✅ Database setup complete."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Export database connection string
echo "🌍 Setting environment variable..."
echo 'export DATABASE_URL="postgresql://username:password@localhost:5432/your_database"' >> ~/.bashrc
source ~/.bashrc

echo "✅ Setup completed successfully! 🚀"
