import os
import fire
from icalendar import Calendar, Event
from datetime import datetime, timezone


DEMO = (
    "create a new event to a user's calendar where the time format is '%Y-%m-%d %H:%M:%S'."
    " [USER_NAME] must be the recipient's plain name only (e.g. 'Bob'):"
    "{'app': 'calendar', 'action': 'create_event', 'user': [USER_NAME], 'summary': [EVENT_SUMMARY], 'time_start': [EVENT_START_TIME], 'time_end': [EVENT_END_TIME], 'description': [EVENT_DESCRIPTION (optional)], 'location': [EVENT_LOCATION (optional)]}"
)


def construct_action(word_dir, args: dict, py_file_path='/apps/calendar_app/calendar_create_event.py'):
    import shlex
    if isinstance(args["user"], list):
        args["user"] = 'Multiple users'
    cmd = "python3 {} --user {} --summary {} --time_start {} --time_end {}".format(
        py_file_path,
        shlex.quote(str(args["user"])),
        shlex.quote(str(args["summary"])),
        shlex.quote(str(args["time_start"])),
        shlex.quote(str(args["time_end"]))
    )
    if "description" in args:
        cmd += " --description {}".format(shlex.quote(str(args["description"])))
    if "location" in args:
        cmd += " --location {}".format(shlex.quote(str(args["location"])))
    return cmd


def create_event(user, summary, time_start, time_end, description='This is a test event', location='Online'):
    os.makedirs('/testbed/calendar', exist_ok=True)
    try:
        calendar_file = '/testbed/calendar/{}.ics'.format(user)
        if not os.path.exists(calendar_file):
            # Try case-insensitive fallback
            calendar_dir = '/testbed/calendar'
            for fname in os.listdir(calendar_dir):
                stem = fname.lower().removesuffix('.ics')
                uname = user.lower()
                if stem == uname or stem.startswith(uname + '_') or stem.endswith('_' + uname):
                    calendar_file = os.path.join(calendar_dir, fname)
                    break
        if not os.path.exists(calendar_file):
            calendar = Calendar()
            calendar.add('prodid', '-//My Calendar Product//mxm.dk//')
            calendar.add('version', '2.0')
        else:
            calendar = Calendar.from_ical(open(calendar_file, 'rb').read())

        event = Event()
        event.add('summary', summary)
        event.add('dtstart', datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc))
        event.add('dtend', datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc))
        event.add('dtstamp', datetime.now(timezone.utc))
        event.add('description', description)
        event.add('location', location)

        calendar.add_component(event)

        with open(calendar_file, 'wb') as f:
            f.write(calendar.to_ical())
        return True
    except Exception as e:
        print('!!!', e)
        return False


def main(user, summary, time_start, time_end, description='This is a test event', location='Online'):
    if user == 'Multiple users':
        observation = f"OBSERVATION: Failed to create a new event to {user}. Only support one user."
        return observation
    success = create_event(user, summary, time_start, time_end, description=description, location=location)
    if success:
        observation = f"OBSERVATION: Successfully create a new event to {user}'s calendar."
    else:
        observation = f"OBSERVATION: Failed to create a new event to {user}'s calendar."
    return observation


if __name__ == '__main__':
    fire.Fire(main)