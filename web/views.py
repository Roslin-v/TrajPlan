from django.shortcuts import render, HttpResponse, redirect

# Create your views here.


def index(request):
    name = 'Roslin'
    print(request.GET)
    print(request.POST)
    # return HttpResponse('Welcome')
    # return redirect('https://www.baidu.com')
    return render(request, 'index.html', {'user': name})
