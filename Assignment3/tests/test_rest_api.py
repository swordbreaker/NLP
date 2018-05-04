from ticketing.api_helper import ApiHelper
import json
import time

helper = ApiHelper()


# Fetch all tickets:
with open('data/ticketingDump.json', 'a') as outfile:
    outfile.write('{')
    for startAt in range(0,96000, 1000):
        outfile.write('"data'+ str(startAt/1000) + '": [')
        tickets = helper.do_get_tickets(str(startAt), str(1000))
        json.dump(tickets, outfile)
        if startAt == 95000:
            outfile.write(']')
        else:
            outfile.write('], ')
        time.sleep(5)
        print(startAt)
    outfile.write('}')


# tickets = helper.do_get_tickets(str(95000), str(10))
# print(json.dumps(tickets))


# print(json.dumps(tickets))
ticket = helper.do_get_ticket('20180406173753758')
print(json.dumps(ticket['messages'],ensure_ascii=False))

categories = helper.do_get_all_categories()
print(json.dumps(categories))

# tickets = helper.do_get_tickets_by_category('VOIP\\/Telefone')
tickets = helper.do_get_tickets_by_category('ict.windisch.support')
print(json.dumps(tickets))
