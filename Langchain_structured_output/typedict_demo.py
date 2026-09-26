from typing import TypedDict

class person(TypedDict):
    
    name = str
    age = int

new_person: person = {'name':'subha','age':27}

print(new_person)