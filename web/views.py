from django.http import HttpResponseRedirect
from django.shortcuts import render, HttpResponse, redirect
from django.urls import reverse
from model.train import predict
from model.algorithm import PlanManager
from model.parser import get_args
from .models import *
from .forms import *
from .response import Response
from hashlib import sha256


def index(request):
    return render(request, 'index.html', Response(200001).res2dict())


def login(request):
    if request.session.get('is_login', None):
        return render(request, 'index.html', Response(200001).res2dict())

    if request.method == 'GET':
        login_form = LoginForm()
        return render(request, 'login.html', Response(200001, login_form).res2dict())

    login_form = LoginForm(request.POST)
    if login_form.is_valid():
        email = str(login_form.cleaned_data.get('email'))
        password = str(login_form.cleaned_data.get('passwd'))
        # 用户存在
        if User.objects.filter(email=email):
            user = User.objects.get(email=email)
            hash_passwd = sha256(password.encode('utf-8')).hexdigest()
            # 密码正确
            if user.password == hash_passwd:
                request.session['is_login'] = True
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                return render(request, 'index.html', Response(200011).res2dict())

    # 用户不存在或密码错误
    return render(request, 'login.html', Response(200010, login_form).res2dict())


def signup(request):
    if request.method == 'GET':
        signup_form = SignupForm()
        return render(request, 'signup.html', Response(200001, signup_form).res2dict())

    signup_form = SignupForm(request.POST)
    if signup_form.is_valid():
        name = str(signup_form.cleaned_data.get('name'))
        email = str(signup_form.cleaned_data.get('email'))
        password = str(signup_form.cleaned_data.get('passwd'))
        # 用户已存在，注册失败
        if User.objects.filter(email=email):
            return render(request, 'signup.html', Response(200020, signup_form).res2dict())

        user = User()
        user.name = name
        user.email = email
        user.password = sha256(password.encode('utf-8')).hexdigest()
        user.save()
        login_form = LoginForm()
        return render(request, 'login.html', Response(200021, login_form).res2dict())

    return render(request, 'signup.html', Response(200020, signup_form).res2dict())


def logout(request):
    if not request.session.get('is_login', None):
        return redirect("../index")
    request.session.flush()
    return redirect("../index")


def diyplan(request):
    # 先判断用户是否登录
    if not request.session.get('is_login', None):
        return render(request, 'plan.html', Response(200000).res2dict())

    spots = Spot.objects.values_list('name', flat=True)
    foods = set(Food.objects.filter().values_list('category', flat=True))
    foods = sorted(foods, key=str.lower)

    if request.method == 'GET':
        return render(request, 'plan.html', Response(200001, {'spots': spots, 'foods': foods}).res2dict())

    # 定制行程
    user_time = int(request.POST.get('user_time'))
    user_budget = int(request.POST.get('user_budget'))
    p_trans = request.POST.get('prefer_trans')
    s_spot = request.POST.getlist('select_spot')
    food_type = request.POST.getlist('food_type')
    if user_time < 1 or user_time > 7 or user_budget < 100:
        return render(request, 'plan.html', Response(200030, {'spots': spots, 'foods': foods}).res2dict())

    if p_trans == '公共交通':
        prefer_trans = 0
    else:
        prefer_trans = 1
    select_spot = []
    for each in s_spot:
        select_spot.append(Spot.objects.get(name=str(each)).id)
    lunch_no = set(Food.objects.values_list('category_id', flat=True))
    for each in food_type:
        c_id = Food.objects.filter(category=str(each)).first().category_id
        if c_id in lunch_no:
            lunch_no.remove(c_id)
    l = [203, 219, 220, 221, 228, 232, 233, 250, 256, 257, 260, 265]
    for each in l:
        lunch_no.add(each)
    args = get_args()
    constraint = {'user-time': user_time * 24,
                  'user-budget': user_budget,
                  'all-time': 0,
                  'all-budget': 0,
                  'prefer-trans': prefer_trans,
                  'select-spot': select_spot,
                  'select-food': [],
                  'lunch-no': list(lunch_no),
                  'dinner-no': list(lunch_no)}

    # 规划并丰富行程
    print(constraint)
    plan_manager = PlanManager(constraint)
    try:
        plan_manager.check_constraint()
        plan_manager.ant_colony()
        if plan_manager.constraint['all-time'] < plan_manager.constraint['user-time'] and plan_manager.constraint['all-budget'] < plan_manager.constraint['user-budget'] / 2:
            plan_manager.plan, plan_manager.constraint = predict(args, 1, plan_manager.plan,
                                                                 plan_manager.constraint)
        plan_manager.improve_plan()
        plan_manager.get_trans()
    except:
        return render(request, 'plan.html', Response(200030, {'spots': spots, 'foods': foods}).res2dict())

    # Todo: 处理plan和trans，让前端可以友好地显示
    return render(request, 'plan.html', Response(200031, {'plan': plan_manager.plan}).res2dict())


