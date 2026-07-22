import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '5555', database = 'pythondb')
mycursor = conn.cursor()

sql = "INSERT INTO student (name, branch, id) VALUES (%s, %s, %s)"
#val= ('John','CSE',56)

#if users want to create multiples values then you can create list
val = [('John','CSE',56),('mike','it',78),('tyson','ME',80)]

#mycursor.executemany(sql,val)
mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount,"record inserted")

