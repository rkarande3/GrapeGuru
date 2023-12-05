
READ ME

To run the app:

1) Make sure all necessary dependencies are downloaded (tensor flow, pandas, etc.)
2) cd server 
3) cd api
4) python model.py (trains the model before you can use app)
5) python app.py (runs backend)
6) cd out of server and into client
7) npm run start (runs frontend)

You can input a description of wine such as "sweet fruity wine with flavors of cherry" to the input field of the app and hit submit. The recommender will pass this into the model and predict which wine type you would like, outputting  the variety as the result.

Note: Try to avoid using special characters and punctuation in your input.
Note: The model does overfit and it is likely you may get the same output multiple times. Significantly different inputs will lead to different outcomes.
