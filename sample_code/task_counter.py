def count_completed(tasks):
    return sum(1 for task in tasks if task["done"])
