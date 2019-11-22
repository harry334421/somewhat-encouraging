a = 1
b = 1

for count in range(99):
    a,b = b,a+b
else:
    print(b)

#result:"573147844013817084101"