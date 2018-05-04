import sys

sys.path.append('../app/Chat_bot/Chat_bot/')

from util.ldap import LDAPHelper

ldap = LDAPHelper()
ldap.get_account_status_edu("yanick.schraner@students.fhnw.ch")
