import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '5555')

if conn.is_connected():
    print("Connection established")
# print(conn)

mycursor = conn.cursor()
mycursor.execute("create database pythondb")
print(mycursor)