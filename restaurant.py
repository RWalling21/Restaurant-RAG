from langchain.pydantic_v1 import BaseModel, Field
from typing import List

class Review(BaseModel):
    reviewer: str = Field(description="The person writing the review")
    review: str = Field(description="The text of the review")

class RestaurantJSON(BaseModel):
    restaurant: str = Field(description="The name of the restaurant")
    phone_number: str = Field(description="The phone number of the restaurant")
    email: str = Field(description="The email of the restaurant")
    dietary_options: List[str] = Field(description="The dietary options the restaurant offers")
    customer_reviews: List[Review] = Field(description="A review of the restaurant with only the reviewer and the review text")