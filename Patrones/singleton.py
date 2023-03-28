#implementación creacional
class Singleton(type):
    '''
    clase que crea una unica instancia de la clase

    _instances es un diccionario que almacena la instancia unica 
    __call__ es el metodo que se ejecuta cuando se crea una instancia de la clase
    
    *args se utiliza para pasar un número variable de argumentos posicionales (es decir, argumentos sin nombre) a una función, y los almacena como una tupla en la variable args.
    *kwargs se utiliza para pasar un número variable de argumentos de palabra clave (es decir, argumentos con nombre) a una función, y los almacena como un diccionario en la variable kwargs.
    '''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]