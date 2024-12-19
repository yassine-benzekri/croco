from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from RAG.rag import EMBSBot

bot = EMBSBot()  # Initialize your bot instance
app = FastAPI()  # Create the FastAPI app

# Define the Pydantic model for request validation
class UserQuery(BaseModel):
    user_query: str

# Define the POST endpoint for chat interactions
@app.post("/chat")
async def chat_endpoint(payload: UserQuery = Body(...)):
    try:
        # Get the response from your bot
        response = await bot.get_response(payload.user_query)
        return {"answer": response}
    except Exception as e:
        # Handle exceptions and return an appropriate error response
        raise HTTPException(status_code=500, detail=str(e))
