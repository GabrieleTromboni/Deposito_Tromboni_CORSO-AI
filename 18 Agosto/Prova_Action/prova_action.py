def main():
x=5
y=10
z=add(x,y)
print("Somma:",z)
def add(a,b):
return a+b
def subtract(a,b):
return a-b
def multiply(a,b):
return a*b
def divide(a,b):
if b==0:
print("Errore: Divisione per zero")
return None
return a/b
main()