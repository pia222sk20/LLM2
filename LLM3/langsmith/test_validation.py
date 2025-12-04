from pydantic import BaseModel, Field
class User(BaseModel):
    name : str = Field(default="무명", description='사용자 이름',min_length=2)
    age : int = Field(default=1, ge=1,description='나이 (0이상) ')

user = User(name='홍길',age=0)
print(user)
print(user.name)

class User2():    
    def __init__(self,name:str, age:int):
        print(name, age)

user2 = User2('홍길동',"10살")

