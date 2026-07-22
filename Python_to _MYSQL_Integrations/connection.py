import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '5555')

if conn.is_connected():
    print("Connection Established")
print(conn)
print(conn.is_connected())
    