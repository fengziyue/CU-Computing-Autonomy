var = 0

def add():
    global var
    var = var + 1

for i in range(50000000):
    add()
print(var)