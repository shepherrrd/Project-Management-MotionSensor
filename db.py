import mysql.connector
import msvcrt
class DatabaseManager:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="pm"
        )
        self.cursor = self.conn.cursor()

    def execute_query(self, query, params=None):
        if params is not None:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        return self.cursor

    def get_setting_value(self):
        query = "SELECT * FROM verified"
        self.execute_query(query)
        result = self.cursor.fetchone()
        if result:
            return bool(result[0])
        else:
            return False

    def close_connection(self):
        self.conn.close()
if __name__ == "__main__":
    # Replace these values with your actual database credentials
    db = DatabaseManager()

    # Example of executing a custom query
    custom_query = "SELECT * FROM verified"
    f = db.get_setting_value()
    print(f)

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'q':
                print("Closing the connection.")
                db.close_connection()
                break