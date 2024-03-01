import copy
import time

from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from model.train import predict
from model.algorithm import PlanManager
from model.parser import get_args
from .models import *
from .forms import *
from .response import Response
from hashlib import sha256


plan_manager = PlanManager()


def index(request):
    spot = Spot.objects.values('name', 'score', 'pic')[:9]
    suggestions = [{'id': 1, 'name': '厦门精华一日游', 'price': 229, 'spot': '鼓浪屿-日光岩-菽庄花园-中山路', 'day': 1},
                   {'id': 2, 'name': '厦门度假三日游', 'price': 639, 'spot': '鼓浪屿-中山路-厦大-环岛路-集美学村', 'day': 3},
                   {'id': 3, 'name': '厦门休闲四日游', 'price': 989, 'spot': '鼓浪屿-中山路-厦大-集美学村-植物园', 'day': 4}]
    return render(request, 'index.html', Response(200001, {'spot': spot, 'suggestion': suggestions}).res2dict())


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

    spots = Spot.objects.values('id', 'name')

    # ========== 编辑行程
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # 评分
        if int(request.POST.get('signal')) == 3:
            if float(request.POST.get('score')) < 0 or float(request.POST.get('score')) > 5:
                return JsonResponse(Response(200100).res2dict())
            plan_manager.user_score = float(request.POST.get('score'))
            return JsonResponse(Response(200101).res2dict())

        old_id = int(request.POST.get('old'))

        # 删除行程
        if int(request.POST.get('signal')) == 2:
            plan_manager.plan_change[old_id] = [2]
            return JsonResponse(Response(200081).res2dict())

        # 替换行程
        new_id = int(request.POST.get('new'))
        new_info = Spot.objects.filter(id=new_id).values('id', 'name', 'price', 'description', 'pic')[0]
        plan_manager.plan_change[old_id] = [1, new_id]

        str_html = "<div class='latest-post-thumb'><img src='/assets/images/" + new_info['pic'] + "' alt='Latest Post'/></div>" \
                   "<div class='latest-post-desc'><h3 class='latest-post-title'>" + new_info['name'] + "</h3>"
        if new_info['price']:
            str_html += ("<span class='price'>门票：" + str(new_info['price']) + "元</span>")
        else:
            str_html += "<span class='price'>门票：免费</span>"
        if new_info['description']:
            str_html += "<span class ='latest-post-meta'>" + new_info['description'] + "</span >"
        str_html += ("<div id='edit" + str(old_id) + "' style='margin-top: 25px;'>"
                    "<select id='changeSelect" + str(old_id) + "' style='width: 200px;'>")
        for s in spots:
            if new_info['id'] == s['id']:
                str_html += ("<option id='" + str(old_id) + str(s['id']) + "' value='" + str(s['id']) + "' "
                            "selected>" + s['name'] + "</option>")
            else:
                str_html += ("<option id='" + str(old_id) + str(s['id']) + "' value='" + str(s['id']) + "'>" +
                            s['name'] + "</option>")
        str_html += ("</select><button id='changeBtn" + str(old_id) + "' class='theme-btn' type='button' "
                      "onclick='changeSpot(" + str(old_id) + ");' style='height: 50px; margin-left: 20px;'>"
                     "替换</button><button id='deleteBtn" + str(old_id) + "' class='theme-btn' "
                    "style='height: 50px; margin-left: 20px;'>删除</button></div></div>")

        return JsonResponse(Response(200071, {'str': str_html}).res2dict())

    # 保存修改的行程
    if request.method == 'POST' and int(request.POST.get('signal', 0)):
        plan_copy = copy.deepcopy(plan_manager)
        try:
            plan_manager.change_plan()
            plan_manager.improve_plan()
            plan_manager.get_plan_print()
            plan_manager.get_trans()
            plan_manager.get_trans_print()
            plan_manager.evaluate()
            return render(request, 'plan.html', Response(200091, {'plan': plan_manager.plan_print,
                                                                  'trans': plan_manager.trans_print,
                                                                  'score': round(plan_manager.score / 20, 1),
                                                                  'days': len(plan_manager.plan),
                                                                  'budget': int(plan_manager.constraint['all-budget']),
                                                                  'spots': spots}).res2dict())
        except:
            plan_manager.callback(plan_copy)
            return render(request, 'plan.html', Response(200090, {'plan': plan_manager.plan_print,
                                                                  'trans': plan_manager.trans_print,
                                                                  'score': round(plan_manager.score / 20, 1),
                                                                  'days': len(plan_manager.plan),
                                                                  'budget': int(plan_manager.constraint['all-budget']),
                                                                  'spots': spots}).res2dict())

    # ========== 返回定制行程的表单
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
    for i in range(49, 55):
        cat_temp.append(category[i])
    cat_double.append(['小吃快餐', cat_temp])

    if request.method == 'GET':
        return render(request, 'plan.html', Response(200001, {'spots': spots, 'foods': cat_double}).res2dict())

    # ========== 定制行程
    user_time = int(request.POST.get('user_time'))
    user_budget = int(request.POST.get('user_budget'))
    prefer_trans = int(request.POST.get('prefer_trans'))
    s_spot = request.POST.getlist('select_spot')
    f_type = request.POST.getlist('food_type')
    select_spot = []
    for each in s_spot:
        select_spot.append(int(each))
    food_type = []
    for each in f_type:
        food_type.append(int(each))
    if user_time < 1 or user_time > 7 or user_budget < 100:
        return render(request, 'plan.html', Response(200030, {'spots': spots, 'foods': cat_double,
                                                              'user_time': user_time, 'user_budget': user_budget,
                                                              'prefer_trans': prefer_trans, 'select_spot': select_spot,
                                                              'food_type': food_type}).res2dict())

    lunch_no = set(Food.objects.values_list('category_id', flat=True))
    for c_id in food_type:
        if c_id in lunch_no:
            lunch_no.remove(c_id)
    lunch_no.add(243)   # 酒吧
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
    start_time = time.time()
    plan_manager.reinitial(constraint)
    try:
        plan_manager.check_constraint()
        plan_manager.ant_colony()
        if plan_manager.constraint['all-time'] < plan_manager.constraint['user-time'] and plan_manager.constraint[
            'all-budget'] < plan_manager.constraint['user-budget'] / 2:
            plan_manager.plan, plan_manager.constraint = predict(args, 1, plan_manager.plan,
                                                                 plan_manager.constraint, plan_manager.expand_day)
        plan_manager.improve_plan()
        plan_manager.get_trans()
        plan_manager.get_plan_print()
        plan_manager.get_trans_print()
        plan_manager.evaluate()
    except:
        return render(request, 'plan.html', Response(200030, {'spots': spots, 'foods': cat_double,
                                                              'user_time': user_time, 'user_budget': user_budget,
                                                              'prefer_trans': prefer_trans, 'select_spot': s_spot,
                                                              'food_type': food_type}).res2dict())

    return render(request, 'plan.html', Response(200031, {'plan': plan_manager.plan_print,
                                                          'trans': plan_manager.trans_print,
                                                          'score': round(plan_manager.score/20, 1),
                                                          'days': len(plan_manager.plan),
                                                          'budget': int(plan_manager.constraint['all-budget']),
                                                          'time': round(time.time() - start_time, 2),
                                                          'spots': spots}).res2dict())


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
    for i in range(49, 55):
        cat_temp.append(category[i])
    cat_double.append(['小吃快餐', cat_temp])

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


def show_suggestion(request):
    if request.method == 'GET':
        s_id = request.GET.get('id', '0')
        if s_id == '1':
            name = '厦门精华一日游'
            days = 1
            budget = 229
            plan = [[1, "鼓浪屿-日光岩-菽庄花园-中山路步行街-八市场", [
                [1, '鼓浪屿', '9:00-16:00', 0, '【鼓浪屿】是世界文化遗产，特色店铺众多，建筑漂亮，生活气息和文艺范儿并重', '鼓浪屿.jpg'],
                [1, '日光岩', '9:00-11:00', 50, '前往【日光岩】，这里是鼓浪屿的制高点，可以俯瞰鼓浪屿及厦门部分市区的景色', '日光岩.jpg'],
                [2, '那私厨｜红砖老别墅餐厅(鼓浪屿店)', 4.9, '闽菜', 100],
                [1, '菽庄花园', '13:00-14:00', 30, '参观【菽庄花园】，菽庄花园在鼓浪屿的地位等同于故宫在北京的地位，很是值得一去，园中还有四十四桥和钢琴博物馆，即可赏景又可听音',
                 '菽庄花园.jpg'],
                [1, '中山路步行街', '15:00-18:00', 0, '前往中国大陆仅有的一条直通大海的商业街【中山路步行街】自由逛吃，去主街附近的小巷中探寻历史厦门和地道人文，去找寻隐匿在小巷中的特色小吃',
                 '中山路步行街.jpg'],
                [1, '八市', '15:00-16:00', 0,
                 '逛吃【第八市场】“菜市场是一座城市的灵魂”，而八市对于厦门而言，就是这样的存在，这里是厦门地道的海鲜市场，不仅藏着很多老牌小吃，海鲜也是非常新鲜，在这里你能品尝到属于厦门的市井味道',
                 '八市.jpg'],
                [2, '乌堂沙茶面(中山路店)', 4.6, '闽菜', 49]]]]
        elif s_id == '2':
            name = '厦门度假三日游'
            days = 3
            budget = 639
            plan = [[1, "鼓浪屿-日光岩-菽庄花园-中山路步行街-八市", [
                [1, '鼓浪屿', '9:00-16:00', 0, '【鼓浪屿】是世界文化遗产，特色店铺众多，建筑漂亮，生活气息和文艺范儿并重', '鼓浪屿.jpg'],
                [1, '日光岩', '9:00-11:00', 50, '前往【日光岩】，这里是鼓浪屿的制高点，可以俯瞰鼓浪屿及厦门部分市区的景色', '日光岩.jpg'],
                [2, '那私厨｜红砖老别墅餐厅(鼓浪屿店)', 4.9, '闽菜', 100],
                [1, '菽庄花园', '13:00-14:00', 30, '参观【菽庄花园】，菽庄花园在鼓浪屿的地位等同于故宫在北京的地位，很是值得一去，园中还有四十四桥和钢琴博物馆，即可赏景又可听音',
                 '菽庄花园.jpg'],
                [1, '中山路步行街', '15:00-18:00', 0, '前往中国大陆仅有的一条直通大海的商业街【中山路步行街】自由逛吃，去主街附近的小巷中探寻历史厦门和地道人文，去找寻隐匿在小巷中的特色小吃',
                 '中山路步行街.jpg'],
                [1, '八市', '15:00-16:00', 0,
                 '逛吃【第八市场】“菜市场是一座城市的灵魂”，而八市对于厦门而言，就是这样的存在，这里是厦门地道的海鲜市场，不仅藏着很多老牌小吃，海鲜也是非常新鲜，在这里你能品尝到属于厦门的市井味道',
                 '八市.jpg'],
                [2, '乌堂沙茶面(中山路店)', 4.6, '闽菜', 49]]],
                    [2, "南普陀寺-厦门大学-胡里山炮台-环岛路", [
                        [1, '南普陀寺', '9:00-11:00', 0, '前往闽南佛教胜地【南普陀寺】依山伴海而建，在这取一把香火，祈一生之福', '南普陀寺.jpg'],
                        [2, '南普陀素菜馆', 4.5, '素食', 70],
                        [1, '厦门大学', '13:00-15:00', 0,
                         '前往参观被誉为“海上花园”的【厦门大学】，校区环抱大海、背依青山，构成了一幅“校在海中 水漾校园”的巨幅画作；芙蓉隧道、上弦场、颂恩楼等是校内知名打卡点（提前三天自行预约）',
                         '厦门大学.jpg'],
                        [1, '胡里山炮台', '16:00-18:00', 23, '到【胡里山炮台】远眺金门大担诸岛，如果有个地方集中了广袤海景、厚重历史和隐约乡愁，那么我想这个地方是胡里山炮台',
                         '胡里山炮台.jpg'],
                        [2, '临家闽南菜(环岛路店)', 4.7, '闽菜', 142],
                        [1, '环岛路', '19:00-21:00', 0, '前往【环岛路】，临海柏油路，葱郁的亚热带树木花草，如会骑自行车，迎着海风骑行游览环岛路也是不错的选择', '环岛路.jpg']]],
                    [3, "集美学村-龙舟池-鳌园-黄厝沙滩", [
                        [1, '集美学村', '9:00-12:00', 0, '参观【集美学村】临海而立，以华侨建筑群而闻名，这里学府聚集，伴随的是美食扎堆，既有浓厚的学识氛围，也有欢快的青年气息',
                         '集美学村.jpg'],
                        [1, '龙舟池', '9:00-10:00', 0, '集美学村的一大亮点，由陈嘉庚先生督建，池畔的亭榭琉璃盖瓦、雕梁画栋，仿佛置身那个时代，奢享悠闲慢时光', '龙舟池.jpg'],
                        [2, '非常台非常泰', 4.7, '台湾菜', 70],
                        [1, '鳌园', '13:00-14:00', 0,
                         '【鳌园】是陈嘉庚先生的陵墓所在地，园内共有666幅栩栩如生的青石雕，还有开国元勋手书题字的解放纪念碑，在园内细细品味人文，而举目望去，海景同样尽收眼底', '鳌园.jpg'],
                        [1, '黄厝沙滩', '15:00-17:00', 0, '到【黄厝沙滩】踩沙踏浪，在树荫下看海，光着脚踩沙，非常惬意', '黄厝沙滩.jpg'],
                        [2, '海渔家餐厅', 5.0, '海鲜', 105]]]]
        elif s_id == '3':
            name = '厦门休闲四日游'
            days = 5
            budget = 989
            plan = [[1, "鼓浪屿-日光岩-菽庄花园-中山路步行街-八市", [
                [1, '鼓浪屿', '9:00-16:00', 0, '【鼓浪屿】是世界文化遗产，特色店铺众多，建筑漂亮，生活气息和文艺范儿并重', '鼓浪屿.jpg'],
                [1, '日光岩', '9:00-11:00', 50, '前往【日光岩】，这里是鼓浪屿的制高点，可以俯瞰鼓浪屿及厦门部分市区的景色', '日光岩.jpg'],
                [2, '那私厨｜红砖老别墅餐厅(鼓浪屿店)', 4.9, '闽菜', 100],
                [1, '菽庄花园', '13:00-14:00', 30, '参观【菽庄花园】，菽庄花园在鼓浪屿的地位等同于故宫在北京的地位，很是值得一去，园中还有四十四桥和钢琴博物馆，即可赏景又可听音',
                 '菽庄花园.jpg'],
                [1, '中山路步行街', '15:00-18:00', 0, '前往中国大陆仅有的一条直通大海的商业街【中山路步行街】自由逛吃，去主街附近的小巷中探寻历史厦门和地道人文，去找寻隐匿在小巷中的特色小吃',
                 '中山路步行街.jpg'],
                [1, '八市', '15:00-16:00', 0,
                 '逛吃【第八市场】“菜市场是一座城市的灵魂”，而八市对于厦门而言，就是这样的存在，这里是厦门地道的海鲜市场，不仅藏着很多老牌小吃，海鲜也是非常新鲜，在这里你能品尝到属于厦门的市井味道',
                 '八市.jpg'],
                [2, '乌堂沙茶面(中山路店)', 4.6, '闽菜', 49]]],
                    [2, "南普陀寺-厦门大学-白城沙滩-环岛路", [
                        [1, '南普陀寺', '9:00-11:00', 0, '前往闽南佛教胜地【南普陀寺】依山伴海而建，在这取一把香火，祈一生之福', '南普陀寺.jpg'],
                        [2, '南普陀素菜馆', 4.5, '素食', 70],
                        [1, '厦门大学', '13:00-15:00', 0,
                         '前往参观被誉为“海上花园”的【厦门大学】，校区环抱大海、背依青山，构成了一幅“校在海中 水漾校园”的巨幅画作；芙蓉隧道、上弦场、颂恩楼等是校内知名打卡点（提前三天自行预约）',
                         '厦门大学.jpg'],
                        [1, '白城沙滩', '16:00-18:00', 0, '漫步【白城沙滩】，迎着海风，在沙滩上肆意奔跑撒欢，或闲坐听风看海', '白城沙滩.jpg'],
                        [2, '临家闽南菜(环岛路店)', 4.7, '闽菜', 142],
                        [1, '环岛路', '19:00-21:00', 0, '前往【环岛路】，临海柏油路，葱郁的亚热带树木花草，如会骑自行车，迎着海风骑行游览环岛路也是不错的选择', '环岛路.jpg']]],
                    [3, "集美学村-龙舟池-鳌园-黄厝沙滩", [
                        [1, '集美学村', '9:00-12:00', 0, '参观【集美学村】临海而立，以华侨建筑群而闻名，这里学府聚集，伴随的是美食扎堆，既有浓厚的学识氛围，也有欢快的青年气息',
                         '集美学村.jpg'],
                        [1, '龙舟池', '9:00-10:00', 0, '集美学村的一大亮点，由陈嘉庚先生督建，池畔的亭榭琉璃盖瓦、雕梁画栋，仿佛置身那个时代，奢享悠闲慢时光', '龙舟池.jpg'],
                        [2, '非常台非常泰', 4.7, '台湾菜', 70],
                        [1, '鳌园', '13:00-14:00', 0,
                         '【鳌园】是陈嘉庚先生的陵墓所在地，园内共有666幅栩栩如生的青石雕，还有开国元勋手书题字的解放纪念碑，在园内细细品味人文，而举目望去，海景同样尽收眼底', '鳌园.jpg'],
                        [1, '黄厝沙滩', '15:00-17:00', 0, '到【黄厝沙滩】踩沙踏浪，在树荫下看海，光着脚踩沙，非常惬意', '黄厝沙滩.jpg'],
                        [2, '海渔家餐厅', 5.0, '海鲜', 105]]],
                    [4, "厦门园林植物园-演武大桥观景平台-钟鼓索道-沙坡尾避风坞", [
                        [1, '厦门园林植物园', '9:00-12:00', 30, '前往【植物园】，欣赏中国金钱松、南洋松等稀有植物，形态各异的多肉植物以及墨西哥风情的仙人掌，拍照胜地，不可错过~', '厦门园林植物园.jpg'],
                        [2, '宴遇(万象城店)', 5.0, '创意菜', 145],
                        [1, '演武大桥观景平台', '14:00-15:00', 0, '【演武大桥观景平台】，桥面标高只有5米，被认为是目前世界上离海平面醉近的桥梁，涨潮的时候，海水几乎与桥底齐平，站在观景平台上，大桥上的川流不息和海面上的船来船往尽收眼底，还能仰望近处的双子塔，远眺对岸的鼓浪屿', '演武大桥观景平台.jpg'],
                        [1, '钟鼓索道', '16:00-17:00', 80, '前往体验【钟鼓索道】，全长1000多米，可以饱览鼓浪屿全景，俯瞰厦门岛，远眺台湾小屿', '钟鼓索道.jpg'],
                        [2, '鲶道居食屋·鹅肝寿司(沙坡尾店)', 4.8, '居酒屋', 118],
                        [1, '沙坡尾避风坞', '19:00-21:00', 0, '前往【沙坡尾】，这里曾是一个老式避风港，如今汇集了一大波文艺小店，聚集了各种独立潮牌/手工艺品/小吃美食/咖啡厅等风格各异的店铺，是一个体验老厦门+文艺小资气息的好去处', '沙坡尾避风坞.jpg']]]]
        else:
            suggestions = [{'id': 1, 'name': '厦门精华一日游', 'price': 229, 'spot': '鼓浪屿-日光岩-菽庄花园-中山路', 'day': 1},
                           {'id': 2, 'name': '厦门度假三日游', 'price': 639, 'spot': '鼓浪屿-中山路-厦大-环岛路-集美学村', 'day': 3},
                           {'id': 3, 'name': '厦门休闲四日游', 'price': 989, 'spot': '鼓浪屿-中山路-厦大-集美学村-植物园', 'day': 4}]
            return render(request, 'suggestion.html', Response(200001, {'suggestion': suggestions}).res2dict())

        return render(request, 'suggestion.html', Response(200061, {'id': int(s_id), 'name': name, 'days': days,
                                                                    'budget': budget, 'plan': plan}).res2dict())


def about(request):
    return render(request, 'about.html', Response(200001).res2dict())
