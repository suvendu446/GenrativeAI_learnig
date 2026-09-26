from pydantic import BaseModel
from typing import Optional ,Emailstr
class Student(BaseModel):

    name:str = 'subha'
    age:Optional[int]
    email:Emailstr

new_student = {'age':27}

Student=Student(**new_student)

print(Student)