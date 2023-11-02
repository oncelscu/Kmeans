import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download("THYAO.IS", start="2022-07-05", end="2023-07-05", period="1d", interval="1h")

number_of_grids = 10
number_of_period = number_of_grids + 1
start_price = 70
end_price = 170

class Position:

    def __init__(self, entry_time, entry_price):
        self.entry_datetime = entry_time
        self.entry_price = entry_price

        self.exit_price = None

        self.price_change = None
        self.exit_datetime = None

    def close_position(self, date, price): 
        self.exit_datetime = date
        self.exit_price = price
        self.price_change = price - self.entry_price

        return self

    def get_profit(self):
        return (self.price_change / self.entry_price) * 100
    
    def get_cumulative_profit(self):
        return sum([position.get_profit() for position in pos_list])
    
    def __repr__(self):
        return f'{self.entry_datetime}({self.entry_price:.2f}) --> {self.exit_datetime}({self.exit_price:.2f}) || {self.get_profit():.3f}'
        
    
def get_grid(price):
    limit_list = np.flip(np.linspace(end_price, start_price, number_of_period))

    for i, limit in enumerate(limit_list):
        if price < limit:
            return i

    return number_of_period

def plot_positions(positions_dict, price_data):
    # Create a new figure and subplot
    fig, ax = plt.subplots()

    # Plot the price data
    ax.plot(price_data.index, price_data['Close'], label='Price')

    # Iterate over positions and plot entry and exit points
    for grid, pos_list in positions_dict.items():
        for position in pos_list:
            entry_time = position.entry_datetime
            entry_price = position.entry_price
            exit_time = position.exit_datetime
            exit_price = position.exit_price

            # Plot entry point
            ax.plot(entry_time, entry_price, 'go', markersize=8, label='Entry')

            # Plot exit point
            ax.plot(exit_time, exit_price, 'ro', markersize=8, label='Exit')
    
    limit_list = np.flip(np.linspace(end_price, start_price, number_of_period))
    
    for limit in limit_list:
        ax.axhline(limit, color='gray', linestyle='--', linewidth=0.5)

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Buy-Sell Points')

    # Show the plot
    plt.show()

if __name__ == '__main__':

    prev_grid = 0
    is_grid_in_pos = [False] * number_of_period

    positions_dict = {i: [] for i in range(number_of_period)}

    for time, row in data.iterrows():
        close_price = row['Close']

        curr_grid = get_grid(close_price)

        if prev_grid > curr_grid:
            if not is_grid_in_pos[curr_grid]:
                positions_dict[curr_grid].append(Position(time, close_price))
                is_grid_in_pos[curr_grid] = True

        if prev_grid < curr_grid:
            if is_grid_in_pos[curr_grid - 2]:
                positions_dict[curr_grid - 2][-1].close_position(time, close_price)
                is_grid_in_pos[curr_grid - 2] = False

        prev_grid = curr_grid

    for open_grid in np.where(is_grid_in_pos)[0]:
        positions_dict[open_grid][-1].close_position(time, close_price)
        is_grid_in_pos[open_grid] = False
    

    for grid, pos_list in positions_dict.items():
        print(grid, pos_list)
        cumulative_profit = sum([position.get_cumulative_profit() for position in pos_list])
        #print("Cumulative Profit:", cumulative_profit)
    
    #cumulative_profit_total = sum([position.get_profit() for pos_list in positions_dict.values() for position in pos_list])
    #print("Total Cumulative Profit:", cumulative_profit_total)
        
    plot_positions(positions_dict, data)