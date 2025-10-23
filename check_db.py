import sqlite3
from pprint import pprint

# Connect to the database
db_path = 'career_survey.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in the database:")
for table in tables:
    print(f"- {table[0]}")

# For each table, show schema and first few rows
for table in tables:
    table_name = table[0]
    print(f"\nTable: {table_name}")
    print("Schema:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    print("\nFirst 5 rows:")
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print("  ", row)
        else:
            print("  No data found")
    except sqlite3.Error as e:
        print(f"  Error reading table: {e}")

conn.close()
