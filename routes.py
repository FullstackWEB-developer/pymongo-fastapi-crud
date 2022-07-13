import asyncio
from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import Product, ProductUpdate

from Product_Category_Prediction.product_category_prediction_one import findCategory, find_multi_Category

router = APIRouter()

@router.post("/", response_description="Create a new product", status_code=status.HTTP_201_CREATED, response_model=Product)
def create_product(request: Request, product: Product = Body(...)):
    product = jsonable_encoder(product)
    new_product = request.app.database["products"].insert_one(product)
    created_product = request.app.database["products"].find_one(
        {"_id": new_product.inserted_id}
    )

    return created_product


@router.get("/", response_description="List all products", response_model=List[Product])
def list_products(request: Request):
    products = list(request.app.database["products"].find(limit=100))
    return products


@router.get("/{id}", response_description="Get a single product by id", response_model=Product)
def find_product(id: str, request: Request):
    if (product := request.app.database["products"].find_one({"_id": id})) is not None:
        return product

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with ID {id} not found")


@router.put("/{id}", response_description="Update a product", response_model=Product)
def update_product(id: str, request: Request, product: ProductUpdate = Body(...)):
    product = {k: v for k, v in product.dict().items() if v is not None}

    if len(product) >= 1:
        update_result = request.app.database["products"].update_one(
            {"_id": id}, {"$set": product}
        )

        if update_result.modified_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with ID {id} not found")

    if (
        existing_product := request.app.database["products"].find_one({"_id": id})
    ) is not None:
        return existing_product

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with ID {id} not found")


@router.delete("/{id}", response_description="Delete a product")
def delete_product(id: str, request: Request, response: Response):
    delete_result = request.app.database["products"].delete_one({"_id": id})

    if delete_result.deleted_count == 1:
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with ID {id} not found")

# get the product name prediction with request.query_params
@router.get("/predict/", response_description="Get a single product by name")
async def predict_product(name: str, request: Request, response: Response):
    # create new function predict_product_name with request.query_params name: productName and return the product name
    # create new function predict_product_name with request.query_params name: productName and return the product name
    # predict_category = asyncio.run(findCategory(name))
    if len(predict_category := find_multi_Category(name)) > 0:
        print(predict_category)
        return predict_category

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with name {name} not found")

# @router.get("/{id}/prediction", response_description="Get the product name prediction", response_model=Product)
# def get_prediction(id: str, request: Request):
#     if (product := request.app.database["products"].find_one({"_id": id})) is not None:
#         return product

#     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with ID {id} not found")
