from controller.controller import Controller


controller = Controller()

result = controller.apply(
    "neuroscale-test",
    1,
    512
)

print(result)
