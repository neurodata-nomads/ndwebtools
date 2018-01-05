import configparser

from django.test import TestCase, RequestFactory
from django.contrib.sessions.middleware import SessionMiddleware

from .boss_remote import BossRemote


class BossRemoteTest(TestCase):

    def setUp(self):
        # Every test needs access to the request factory.
        self.factory = RequestFactory()

        boss_config_file = 'neurodata.cfg'
        config = configparser.ConfigParser()
        config.read(boss_config_file)
        self.token = config['Default']['token']

        request = self.factory.get('/index')

        add_session_to_request(request)

        request.session['access_token'] = self.token
        self.boss_remote = BossRemote(request, auth_type='Token')

    def test_create_boss_remote(self):
        assert self.boss_remote.boss_url == 'https://api.boss.neurodata.io/v1/'
        assert self.boss_remote.auth_type == 'Token'

    def test_get_exp_info(self):
        coll = 'ben_dev'
        exp = 'test_render'
        exp_info = self.boss_remote.get_exp_info(coll, exp)
        assert exp_info['creator'] == 'benfalk'
        assert exp_info['hierarchy_method'] == 'isotropic'
        assert 'x_stop' not in exp_info

    def test_get_ch_info(self):
        coll = 'ben_dev'
        exp = 'test_render'
        ch = 'image_test_20171205-230223'
        ch_info = self.boss_remote.get_ch_info(coll, exp, ch)
        assert ch_info['datatype'] == 'uint8'
        assert ch_info['type'] == 'image'

    def test_get_coordinate_frame(self):
        coll = 'ben_dev'
        exp = 'test_render'
        coord_frame_name = self.boss_remote.get_coordinate_frame(coll, exp)
        assert coord_frame_name['name'] == 'ben_dev_test_render'
        assert coord_frame_name['voxel_unit'] == 'micrometers'


def add_session_to_request(request):
    """Annotate a request object with a session"""
    middleware = SessionMiddleware()
    middleware.process_request(request)
    request.session.save()
