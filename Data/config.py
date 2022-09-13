import pymysql


class Base(object):
    def __init__(self, database):
        self.engine = pymysql.connect(host='localhost',
                                      user='root',
                                      passwd='')
        self.database = database

    def select_query(self, query):
        cursor = self.engine.cursor()
        cursor.execute(query)
        res = [r for r in cursor.fetchall()]
        self.engine.commit()
        return res

    def multi_query(self, query, data):
        cursor = self.engine.cursor()
        cursor.executemany(query, data)
        res = [r for r in cursor.fetchall()]
        self.engine.commit()
        return res
