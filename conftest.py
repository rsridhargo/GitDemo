import pytest

#adding comment for git for practicing
#first comment
<<<<<<< HEAD
#second comment
#third comments
=======
#second comments
>>>>>>> 9b746aff0fbafbe288f269d013719a26bfb4f8ef

@pytest.fixture(scope="class")
def setup():
    print("I will be executing first")
    yield
    print(" I will execute last")


@pytest.fixture()
def dataLoad():
    print("user profile data is being created")
    return ["Rahul S","Shetty","rahulshettyacademy.com"]


@pytest.fixture(params=[("chrome","Rahul","shetty"), ("Firefox","shetty"), ("IE","SS")])
def crossBrowser(request):
    return request.param
