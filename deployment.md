# Notes on deployment

### Many of these notes on deployment were taken from the tutorial on nginx and uWSGI: https://uwsgi-docs.readthedocs.io/en/latest/tutorials/Django_and_nginx.html

- Create a virtual environment (python 3)
- Clone the repo
- pip install requirements.txt
- create local_settings.py file
    - SECRET KEY
    - DEBUG
    - ALLOWED_HOSTS
    - auth_uri
    - cliend_id
    - public_uri
- python manage.py migrate
- pip install uwsgi (if not installed systemwide already)
- install nginx (if not already installed)
- drop the uwsgi_params file in your local path: https://github.com/nginx/nginx/blob/master/conf/uwsgi_params
- create mysite_nginx.conf file
    - mysite_nginx.conf
        ```apacheconf
        # mysite_nginx.conf
        # the upstream component nginx needs to connect to
        upstream django {
            server unix:///home/ubuntu/uwsgi-ndwebtools/ndwebtools/mysite.sock; # for a file socket
            # server 127.0.0.1:8001; # for a web port socket (we'll use this first)
        }

        # configuration of the server
        server {
            # the port your site will be served on
            listen      8001;
            # the domain name it will serve for
            server_name ben-dev.neurodata.io; # substitute your machine's IP address or FQDN
            charset     utf-8;

            # max upload size
            client_max_body_size 75M;   # adjust to taste

            # Django media
            location /media  {
                alias /home/ubuntu/uwsgi-ndwebtools/ndwebtools/media;  # your Django project's media files - amend as required
            }

            location /static {
                alias /home/ubuntu/uwsgi-ndwebtools/ndwebtools/static; # your Django project's static files - amend as required
            }

            # Finally, send all non-media requests to the Django server.
            location / {
                uwsgi_pass  django;
                include     /home/ubuntu/uwsgi-ndwebtools/ndwebtools/uwsgi_params; # the uwsgi_params file you installed
            }
        }
        ```
- make symbolic link
    - `sudo ln -s /home/ubuntu/uwsgi-ndwebtools/ndwebtools/mysite_nginx.conf /etc/nginx/sites-enabled/`
- create uwsgi.ini file
    ```ini
    # mysite_uwsgi.ini file
    [uwsgi]

    #plugins = python3

    # Django-related settings
    # the base directory (full path)
    chdir           = /home/ubuntu/uwsgi-ndwebtools/ndwebtools
    # Django's wsgi file
    module          = mysite.wsgi
    # the virtualenv (full path)
    home            = /home/ubuntu/uwsgi-ndwebtools

    # process-related settings
    # master
    master          = true
    # maximum number of worker processes
    processes       = 4
    # the socket (use the full path to be safe
    socket          = /home/ubuntu/uwsgi-ndwebtools/ndwebtools/mysite.sock
    # ... with appropriate permissions - may be needed
    chmod-socket    = 666
    # clear environment on exit
    vacuum          = true
    ```
- restart the nginx server
    - sudo systemctl restart nginx
- run uwsgi
    - uwsgi --ini mysite_uwsgi.ini

site should be responsive at this point


#### Next steps are to make it server ready (autostarting, systemwide installs)
- systemwide install
    - `sudo apt install uwsgi uwsgi-plugins-python3`
- deploy using systemd and `/etc/uwsgi/`
