import sqlite3
from tabulate import tabulate

def dump_db(db_path="images.db"):
    """Dumps the contents of the images.db database to the console in a tabular format."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(images)")
        columns = [col[1] for col in cursor.fetchall()]

        # Get data
        cursor.execute("SELECT image_id, file_name, file_path, width, height, captions, metadata FROM images")
        data = cursor.fetchall()

        # Print table header
        print(tabulate(columns, headers="keys", tablefmt="pretty"))

        # Print table rows
        print(tabulate(data, headers=columns, tablefmt="pretty"))

        conn.close()

    except sqlite3.Error as e:
        print(f"Error accessing database: {e}")
        print("Please ensure the 'images.db' database exists and is accessible.")
    except ImportError:
        print("Error: The 'tabulate' library is not installed.")
        print("Please install it using: pip install tabulate")

if __name__ == "__main__":
    dump_db()