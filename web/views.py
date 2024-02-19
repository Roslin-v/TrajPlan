import math

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


plan_manager = PlanManager()


def index(request):
    spot = Spot.objects.values('name', 'score')[:9]
    return render(request, 'index.html', Response(200001, {'spot': spot}).res2dict())


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
    # 考虑类别，排除203冰激淋 219酒吧 220居酒屋 221咖啡店 228零食 232 233面包 250卤味 256西式快餐 257甜点 260小吃 265饮品
    l = [250, 254, 255, 257, 259, 260, 261, 262, 263, 264, 265, 243]
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
    plan_manager.reinitial(constraint)
    try:
        plan_manager.check_constraint()
        plan_manager.ant_colony()
        if plan_manager.constraint['all-time'] < plan_manager.constraint['user-time'] and plan_manager.constraint['all-budget'] < plan_manager.constraint['user-budget'] / 2:
            plan_manager.plan, plan_manager.constraint = predict(args, 1, plan_manager.plan,
                                                                 plan_manager.constraint)
        plan_manager.improve_plan()
        plan_manager.get_trans()
        plan_manager.get_plan_print()
        plan_manager.get_trans_print()
        plan_manager.evaluate()
    except:
        return render(request, 'plan.html', Response(200030, {'spots': spots, 'foods': foods}).res2dict())

    return render(request, 'plan.html', Response(200031, {'plan': plan_manager.plan_print,
                                                          'trans': plan_manager.trans_print,
                                                          'score': round(plan_manager.score/20, 1),
                                                          'days': len(plan_manager.plan),
                                                          'budget': int(plan_manager.constraint['all-budget'])}).res2dict())


def show_spot(request):
    code = 200001

    if request.method == 'POST':
        search_spot = request.POST.get('search')
        if search_spot:
            try:
                search_result = Spot.objects.filter(name__icontains=search_spot).values('name', 'score', 'price',
                                                                                        'description', 'pic')
                return render(request, 'spot.html',
                              Response(200041, {'spot': search_result, 'search': search_spot}).res2dict())
            except:
                code = 200040

    result = Spot.objects.values('name', 'score', 'price', 'description', 'pic')

    return render(request, 'spot.html', Response(code, {'spot': result[:9]}).res2dict())


def show_food(request):
    code = 200001
    category = Food.objects.values('category_id', 'category').distinct().order_by('category_id')
    cat_double = []
    cat_temp = []
    cat_double.append(['厦门特色', [category[0]]])
    for i in range(1, 3):
        cat_temp.append(category[i])
    cat_double.append(['海鲜', cat_temp])
    cat_temp = []
    for i in range(3, 19):
        cat_temp.append(category[i])
    cat_double.append(['地方菜系', cat_temp])
    cat_temp = []
    for i in range(19, 34):
        cat_temp.append(category[i])
    cat_double.append(['异域料理', cat_temp])
    cat_temp = []
    for i in range(34, 39):
        cat_temp.append(category[i])
    cat_double.append(['火锅', cat_temp])
    cat_temp = []
    for i in range(39, 42):
        cat_temp.append(category[i])
    cat_double.append(['烧烤', cat_temp])
    cat_temp = []
    for i in range(42, 49):
        cat_temp.append(category[i])
    cat_double.append(['其他', cat_temp])
    cat_temp = []
    for i in range(49, 59):
        cat_temp.append(category[i])
    cat_double.append(['小吃快餐', cat_temp])
    cat_temp = []
    for i in range(59, 61):
        cat_temp.append(category[i])
    cat_double.append(['饮品', cat_temp])
    cat_temp = []
    for i in range(61, 66):
        cat_temp.append(category[i])
    cat_double.append(['面包甜品', cat_temp])

    if request.method == 'POST':
        keyword = request.POST.get('keyword')
        category_id = int(request.POST.get('state'))
        min_price = int(request.POST.get('min'))
        max_price = int(request.POST.get('max'))
        search_result = []
        result = []
        try:
            if keyword and category_id != 200:
                search_result = Food.objects.filter(name__icontains=keyword, category_id=category_id).values('name', 'score', 'price', 'category_id')
            elif keyword:
                search_result = Food.objects.filter(name__icontains=keyword).values('name', 'score', 'price', 'category_id')
            elif category_id != 200:
                search_result = Food.objects.filter(category_id=category_id).values('name', 'score', 'price', 'category_id')
            if search_result:
                for each in search_result:
                    if min_price <= each['price'] <= max_price:
                        result.append(each)
                if result:
                    return render(request, 'food.html', Response(200051, {'food': result, 'counts': len(result),
                                                                          'keyword': keyword, 'state': category_id,
                                                                          'min': min_price, 'max': max_price,
                                                                          'category': cat_double}).res2dict())
                else:
                    code = 200050
        except:
            code = 200050

    result = []
    for cat in category:
        f = Food.objects.filter(category_id=cat['category_id']).values('name', 'score', 'price', 'category_id')
        if len(f) > 2:
            f = f[:2]
        result += f

    return render(request, 'food.html', Response(code, {'food': result, 'category': cat_double}).res2dict())
