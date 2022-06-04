import os


class Analysis:
    
    def __init__(self):
        pass

    def name(self):
        return "Analysis"

    def run(self, args):
        os.makedirs(self.result_dir(), exist_ok=True)
        self.steps(args)
        self.write(args)

    def steps(self, args):
        pass

    def write(self, args):
        pass
    
    def result_dir(self):
        return os.sep.join([__file__.replace(".py", ""), self.name()])
