# NBA_Good_Player
Conseiller des investisseurs cherchant à capitaliser sur de futurs talents de la NBA.



# Screenshots

![image](https://drive.google.com/uc?export=view&id=1qlCqYreKC2KMJdW-O82myf3cYwPGko9E)

# Install environment
```python
conda create -n env python=3.9 ipython ipykernel  
pip install django
pip install numpy
pip install pandas

```

# Run the application player

```
cd ./NBA_Good_Player
python manage.py runserver
```
You’ll see the following output on the command line:
```
Performing system checks...

System check identified no issues (0 silenced).

You have unapplied migrations; your app may not work properly until they are applied.
Run 'python manage.py migrate' to apply them.

September 25, 2022 - 15:50:53
Django version 4.1, using settings 'mysite.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

Now that the server’s running, visit http://127.0.0.1:8000/ with your web browser.
