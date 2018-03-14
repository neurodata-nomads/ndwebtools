## Install ndwebtools
- Create a virtual environment (python 3)
    - `virtualenv ndwebtools_env -p /usr/bin/python3`
- Clone ndwebtools into virtual environment path
- Install requirements of ndwebtools
    - `pip install -r requirements.txt`
- create `local_settings.py` file inside mysite/ with the following values:
    - SECRET KEY
    - DEBUG (True/False)
    - ALLOWED_HOSTS (`['*']`)
    - auth_uri
    - cliend_id
    - public_uri
- `python manage.py makemigrations bossoidc`
- `python manage.py migrate`
- create `neurodata.cfg` in `./synaptogram`
    - `pip install intern` for now

### Test to see if site works
- `python manage.py runserver 0:8080`


