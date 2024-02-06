from django import forms


class LoginForm(forms.Form):
    email = forms.CharField(label="邮箱", max_length=20, widget=forms.TextInput(attrs={'class': 'form-control'}))
    passwd = forms.CharField(label="密码", max_length=50, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
