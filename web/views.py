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
    return render(request, 'index.html', Response(200000).res2dict())


def login(request):
    if request.session.get('is_login', None):
        return HttpResponseRedirect(reverse('index'))

    if request.method == 'GET':
        login_form = LoginForm()
        return render(request, 'login.html', Response(200000, login_form).res2dict())

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
        return render(request, 'signup.html', Response(200000, signup_form).res2dict())

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
    if request.method == 'GET':
        return render(request, 'plan.html')

    # ========== 定制行程
    args = get_args()
    constraint = {'user-time': 48,
                  'user-budget': 1000,
                  'all-time': 0,
                  'all-budget': 0,
                  'prefer-trans': 0,
                  'select-spot': [10001, 10002, 10003, 10005, 10006, 10007, 10008, 10009, 10041],
                  'select-food': [],
                  'lunch-no': [203, 219, 220, 221, 228, 232, 233, 250, 256, 257, 260, 265],
                  'dinner-no': [203, 221, 228, 232, 233, 250, 256, 257, 260, 265]}
    plan_manager = PlanManager(constraint)
    plan_manager.check_constraint()
    plan_manager.ant_colony()
    if plan_manager.constraint['all-time'] < plan_manager.constraint['user-time'] and plan_manager.constraint[
        'all-budget'] < plan_manager.constraint['user-budget'] / 2:
        plan_manager.plan, plan_manager.constraint = predict(args, 1, plan_manager.plan, plan_manager.constraint)
    plan_manager.improve_plan()
    plan_manager.get_trans()

    # ========== 构造返回参数

    return render(request, 'plan.html')
