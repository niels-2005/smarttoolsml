from pydantic import BaseModel 


data = {
    "name": "Murthy",
    "age": "28",
    "course": "MLOps Bootcamp",
    "ratings": [4, 4, 4, "4", "5"]
}


class Instructor(BaseModel):
    name: str 
    age: int
    course: str
    ratings: list[int] = []


user = Instructor(**data)


print(f"Found a Instructor: {user}")


# pydantic is very useful for reparsing data in right format
# example: client send str but the model needs int -> pydantic