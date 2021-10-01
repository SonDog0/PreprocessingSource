def get_todo_fix_asin(logfile):

    file = open(logfile,'r', encoding='utf-8')
    lines = file.readlines()

    todo_fix_asin = []
    for idx, line in enumerate(lines):
        list_line = line.split(' ')

        # TODO: fix logfile format and comfile pattern
        # presume all asin start 'B'
        r = re.compile(r"B.")
        filtered_list = list(filter(r.match, list_line))

        # check empty list
        if not filtered_list:
            print(f'line {idx} is not have ASIN')
            continue
        else:
            # else append to todo_fix_list
            todo_fix_asin.append(filtered_list[0])

    return todo_fix_asin
