from django.shortcuts import render, HttpResponse, redirect
from .models import *
from .forms import *
from .response import Response
from hashlib import sha256


def index(request):
    return render(request, 'index.html', Response(200000).res2dict())


def login(request):
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
                return render(request, 'index.html', Response(200011, user.name).res2dict())

    # 用户不存在或密码错误
    return render(request, 'login.html', Response(200010, login_form).res2dict())
