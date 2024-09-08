from fastapi import FastAPI 


# define app
app = FastAPI() 


# get operation with decorator
@app.get("/") 
async def root():
    return {"message":"Hello World from FASTAPI"}


# get operation with decorator
@app.get("/demo")
def demo_func():
    return {"message":"This is output from demo function"}


# post operation with decorator
@app.post("/post_demo") 
def demo_post():
    return {"message":"This is output from post demo function"}


@app.update("/update_demo")
def demo_update():
    return {"message": "This is output from update demo function"}


@app.delete("/delete_demo")
def demo_delete():
    return {"message": "This is output from delete demo function"}