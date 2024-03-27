# TrajPlan

<p align=right>Copyright@ Roslin V.</p>

## About

This is an web application about trip trajectory planning based on user constraints. In order to have a rough overview about this project, you can  refer to `'Instructions'- 'Frontend'`and visit [鼓浪听涛-主页](http://101.200.154.178/index/) for a head start.

The technical route is as follows:

1. Get data

   Get data about spot, food, transportation and trajectory from travel websites and map API.

2. POI embedding

   Embed features of spot, user, time etc. to enrich the context of our model.

3. Satisfy user constraints

   Ask a user to input his/her constraints, such as spots wanted, time limit, budget limit etc. Satisfy them by deleting less important spot(s) using knapsack algorithm.

4. Plan trajectory route

   Plan a route using ant colony algorithm. Divide the route into daily plan according to spot constraints such as recommended visiting time.

5. Recommend other spots and foods

   1. Recommend other spots: Predict next POI using trained neural network.
   2. Recommend foods: Recommend restaurants according to their distances from spots, scores, comments and categories.

6. Get transportation suggestions

   After planning, get transportation suggestions using AMap API. Present them according to reality constraints and users' choices.

## Technology Stack

- Language: Python
- Web Framework: Django 
- Database: MySQL
- Frontend Framework: Bootstrap

## Content

```
|- Readme.md			// help
|- assets					// static files used in the web application
|	|- css
|	|- fonts
|	|- images
|	|- js
|	|- vendor
|- config					// Django project settings etc.
|	|- __init__.py
|	|- asgi.py
|	|- settings.py
|	|- urls.py
|	|- wsgi.py
|- data						// data used in model training
|- model					// algorithms and models used in trajectory plan
|	|- algorithm		// route planning, recommending etc.
|	|- data_proces.py	// process data for model training
|	|- main.py
|	|- model.py			// model layers
|	|- parser.py		// parse argument
|	|- train.py			// train and predict next POI
|- output					// model training output
|- templates			// web pages
|- web						// Django application
|	|- __init.py
|	|- admin.py
|	|- apps.py
|	|- forms.py			// process forms in html
|	|- models.py		// models from database
|	|- response.py	// response to frontend
|	|- views.py			// process requests from frontend
```

## Instructions

### Backend

You need to finish training the model and test functions of backend before using frontend pages. Please check `main.py` for specific steps. Run method `train()` first to get a best trained model.  The result will be saved in `../output/`. Then, annotate it and run the rest codes. You will get a plan, contained spots and foods, and transportation suggestions in terminal. Examples are as follows:

```
---------- Original Plan ----------
>>> Day 1
鼓浪屿 : 9:00-16:00 
厦门海底世界 : 9:00-12:00 门票: 107.0 元
日光岩 : 12:00-14:00 门票: 50.0 元
中山路步行街 : 16:30-19:30 
>>> Day 2
厦门大学 : 9:00-11:00 
曾厝垵 : 11:30-14:30 
胡里山炮台 : 15:00-17:00 门票: 23.0 元
>>> Day 3
环岛路 : 9:00-12:00 
厦门园林植物园 : 12:30-15:30 门票: 30.0 元
Total time:	 63.5 小时
Total fee:	 210.0 元
Score: 79.59/100
---------- Improved Plan ----------
>>> Day 1
鼓浪屿 : 9:00-16:00 
厦门海底世界 : 9:00-12:00 门票: 107.0 元
那私厨｜红砖老别墅餐厅(鼓浪屿店) : 评分 4.9 (闽菜) 人均: 100 元
日光岩 : 13:00-15:00 门票: 50.0 元
中山路步行街 : 17:30-20:30 
醉得意(厦禾店) : 评分 4.8 (其他) 人均: 55 元
>>> Day 2
环岛路 : 9:00-12:00 
周麻婆(中闽百汇店) : 评分 3.8 (川菜) 人均: 49 元
厦门园林植物园 : 13:30-16:30 门票: 30.0 元
云上厦门观光厅 : 17:30-18:30 门票: 158.0 元
南普陀素菜馆 : 评分 4.5 (素食) 人均: 70 元
白城沙滩 : 19:30-21:30 
>>> Day 3
厦门大学 : 9:00-11:00 
金家港海鲜大排档(中山路店) : 评分 4.9 (海鲜) 人均: 145 元
曾厝垵 : 12:30-15:30 
胡里山炮台 : 16:00-18:00 门票: 23.0 元
上青本港海鲜(后江埭路店) : 评分 4.7 (海鲜) 人均: 204 元
白鹭洲公园 : 19:00-21:00 
Total time:	 69.0 小时
Total fee:	 1027.0 元
Score: 93.50/100
Improved by: 17.47%
```

Note: Remember to change arguments in `parser.py` before running codes.

### Frontend

There are ten pages in the frontend. A brief introduction is as follows:

```
|- index.html		// homepage
|- signup.html		// sign up here
|- login.html		// log in here
|- spot.html		// show and search spot
|- food.html		// show and search food
|- suggestion.html	// show trip route suggestion
|- plan.html		// DIY a plan (main function)
|- favorite.html	// show your favorite plans
|- contact.html		// contact us here
|- about.html		// about us
```

#### signup.html

Just to remind you that you can submit a false name and email when signing up our website. This is just an experimental application. Don't worry about the security of your personal information.

#### plan.html

1. Input constraints

   In here, you can input constraints to DIY your plan. Please note a few limits if you want it to respond successfully. 

   1. Day: Must between 1-7.
   2. Limit: Must be larger than 100. If your limit is larger than 100, but still too little considering your travel days, the planning will fail.
   3. Spot & Food: Support multiple select. Need at least one.

2. Change the Plan

   1. Having successfully get your plan, you can replace or delete some of the spots if you are not satisfied with the result. 
   2. Change and save the plan. Please make sure your changes are reasonable. For example, you can't choose the same spot twice. Also, you can't visit a museum in the middle of the night.

3. Favorite the Plan

   1. If you are satisfied with plan, you can score the plan and favorite it (by clicking the heart). 
   2. By doing so, you can see your plan in `favorite.html` again. And we will add this plan into our trajectory dataset for improving our model.
   3. Of course you can unfavorite your plan, but if you do that, you won't be able to see it again unless you DIY a new plan.

#### contact.html

I write this to invite all users to write comments to me. You can report bugs, offer suggestions, or simply praise me for my work. I would be very grateful for whatever opinions you submit. You can submit comment anonymously,  or leave your name or contact information so I could thank you and send you a feedback. 
