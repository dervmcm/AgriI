class Records:
    def __init__(self):
        self.customers = {}
        self.movies = []
        self.tickets = []
        self.run()

    def add_customer(self, id, name, discount=0, threshold=0):
        self.customers[id] = {
            'id': id,
            'name': name,
            'discount': discount,
            'threshold': threshold
        }

    def read_customers(self, filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = line.split(", ")
                    if data[0].startswith("C"):
                        self.add_customer(data[0], data[1])
                    elif data[0].startswith("F"):
                        self.add_customer(data[0], data[1], float(data[2]))
                    elif data[0].startswith("S"):
                        self.add_customer(data[0], data[1], float(data[2]), float(data[3]))

    def read_movies(self, filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = line.split(", ")
                    self.movies.append({"id": data[0], "name": data[1], "seats": int(data[2])})

    def read_tickets(self, filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = line.split(", ")
                    self.tickets.append({"id": data[0], "name": data[1], "price": float(data[2])})

    def find_customer(self, search_value):
        for customer in self.customers.values():
            if search_value == customer["id"] or search_value == customer["name"]:
                return customer
        return None

    def find_movie(self, search_value):
        for movie in self.movies:
            if search_value in (movie["id"], movie["name"]):
                return movie
        return None

    def find_ticket(self, search_value):
        for ticket in self.tickets:
            if search_value in (ticket["id"], ticket["name"]):
                return ticket
        return None

    def display_customers(self):
        print("Customer ID\tCustomer Name\tDiscount\tThreshold")
        for customer in self.customers:
            #print(self.customers[customer]['id'])
            if customer.startswith("C"):
                print(f"{self.customers[customer]['id']}\t{self.customers[customer]['name']}\t\t\t\t\t")
            elif customer.startswith("F"):
                print(f"{self.customers[customer]['id']}\t{self.customers[customer]['name']}\t\t{self.customers[customer]['discount']:.2f}\t\t")
            elif customer.startswith("S"):
                print(f"{self.customers[customer]['id']}\t{self.customers[customer]['name']}\t\t{self.customers[customer]['discount']:.2f}\t\t{self.customers[customer]['threshold']:.2f}")

    def display_movies(self):
        print("Movie ID\tMovie Name\t\tAvailable Seats")
        for movie in self.movies:
            print(f"{movie['id']}\t{movie['name']}\t\t\t\t{movie['seats']}")

    def display_tickets(self):
        print("Ticket ID\tTicket Name\t\tUnit Price")
        for ticket in self.tickets:
            print(f"{ticket['id']}\t{ticket['name']}\t\t\t\t{ticket['price']:.2f}")

    def run(self):
        # read data files
        try:
            self.read_customers("customers.txt")
            #self.read_movies("movies.txt")
            #self.read_tickets("tickets.txt")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
    def getnextgen(self,holder,type):
        number = len(holder)
        newnumber = number + 1
        newid = None
        while True:
            newid = type + str(newnumber)
            for i in holder:
                if i == newid:
                    newnumber += 1
                    return(type + str(newnumber))
            break
        return(newid)
    
    def getnextinput(self,input):
        holder = []        
        for j in record.customers:
            for i in j:
                if i == input:
                    holder.append(j)
        return(self.getnextgen(holder,input))
    

        

if __name__ == "__main__":
    record = Records()
    record.read_customers("customers.txt")
    #record.display_customers()
    #record.add_customer(name ="shit")

    '''for i in record.customers:
        print(record.customers[i])'''
    '''genunit = input("enter type")
    if genunit == "c":
        text = record.getnextgen(cholder, "C")
        cholder.append(text)
    elif genunit == "s":
        text = record.getnextgen(sholder, "S")
        sholder.append(text)
    else:
        text = record.getnextgen(fholder, "F")
        fholder.append(text)'''

    for i in range(10):
        item = record.getnextinput("F")
        record.add_customer(item,"BOB")


    record.display_customers()