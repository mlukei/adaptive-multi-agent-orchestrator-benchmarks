import os
import fire
from icalendar import Calendar
from datetime import datetime


DEMO = (
    "delete an event from a user's calendar given the event summary:"
    "{'app': 'calendar', 'action': 'delete_event', 'user': [USER_NAME], 'summary': [EVENT_SUMMARY]}"
)


def construct_action(word_dir, args: dict, py_file_path='/apps/calendar_app/calendar_delete_event.py'):
    import shlex
    return f'python3 {py_file_path} --user {shlex.quote(str(args["user"]))} --summary {shlex.quote(str(args["summary"]))}'


def delete_event(user, summary):
    try:
        calendar_file = '/testbed/calendar/{}.ics'.format(user)
        if not os.path.exists(calendar_file):
            calendar_dir = '/testbed/calendar'
            if os.path.isdir(calendar_dir):
                for fname in os.listdir(calendar_dir):
                    stem = fname.lower().removesuffix('.ics')
                    uname = user.lower()
                    if stem == uname or stem.startswith(uname + '_') or stem.endswith('_' + uname):
                        calendar_file = os.path.join(calendar_dir, fname)
                        break
        with open(calendar_file, 'rb') as f:
            calendar = Calendar.from_ical(f.read())

        for component in calendar.walk():
            if component.name == "VEVENT":
                if component.get('summary') == summary:
                    calendar.subcomponents.remove(component)
                    break
        with open(calendar_file, 'wb') as f:
            f.write(calendar.to_ical())
        return True
    except:
        return False


def main(user, summary):
    success = delete_event(user, summary)
    if success:
        observation = f"OBSERVATION: Successfully delete an event named {summary} from {user}'s calendar."
    else:
        observation = f"OBSERVATION: Failed to delete an event named {summary} from {user}'s calendar."
    return observation


if __name__ == '__main__':
    fire.Fire(main)