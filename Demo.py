from Container import Container

con = Container()
con.set(["a", "b"], [1, 2])
con.set("a", 3, ["train", "valid", "test"])
con.set("a", 4, ["train", "valid", "test"], ["child"])
con.set("a", 5, ["train", "valid", "test"], ["child", "grandchild"])
print(con.to_string())
