import os
import curses
import torch
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

def load_agent(env, model_path):
    agent = DQNAgent(env)
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0 
    return agent

def select_model_with_curses():
    model_dir = "./models"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found!")
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")

    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)

        current_row = 0
        page_start = 0
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            header_lines = 2
            footer_lines = 2
            items_per_page = height - header_lines - footer_lines
            page_end = page_start + items_per_page
            
            items_per_page = max(items_per_page, 1)
            page_end = min(page_end, len(model_files))
            
            header = " Select a model (↑/↓: Navigate, Enter: Select, Q: Quit) "
            header = header.center(width)
            stdscr.addstr(0, 0, header, curses.color_pair(1))
            
            for idx in range(page_start, page_end):
                if idx >= len(model_files):
                    break
                
                file = model_files[idx]
                display_idx = idx - page_start
                x = (width - len(file)) // 2
                
                if current_row == idx:
                    stdscr.addstr(display_idx + header_lines, x, file, curses.color_pair(2))
                else:
                    stdscr.addstr(display_idx + header_lines, x, file, curses.color_pair(1))

            footer = f" Page {page_start//items_per_page + 1}/{(len(model_files)-1)//items_per_page + 1} "
            footer += "| ↑/↓: More |" if len(model_files) > items_per_page else ""
            stdscr.addstr(height-2, 0, footer.center(width), curses.color_pair(3))
            
            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_UP:
                if current_row > 0:
                    current_row -= 1
                    if current_row < page_start:
                        page_start = max(page_start - items_per_page, 0)
            elif key == curses.KEY_DOWN:
                if current_row < len(model_files) - 1:
                    current_row += 1
                    if current_row >= page_end:
                        page_start = min(page_start + items_per_page, len(model_files) - items_per_page)
            elif key == curses.KEY_PPAGE:  
                current_row = max(current_row - items_per_page, 0)
                page_start = max(page_start - items_per_page, 0)
            elif key == curses.KEY_NPAGE:  
                current_row = min(current_row + items_per_page, len(model_files) - 1)
                page_start = min(page_start + items_per_page, len(model_files) - items_per_page)
            elif key == curses.KEY_ENTER or key in [10, 13]:
                break
            elif key == ord('q') or key == ord('Q'):
                raise KeyboardInterrupt("User exited model selection")

        return os.path.join(model_dir, model_files[current_row])
    finally:
        curses.echo()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.endwin()

def play_game(env, agent):
    clock = pygame.time.Clock()
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        clock.tick(60)

    print(f"\nFinal state: {state}")
    print(f"Total reward: {reward}")
    env.close()

if __name__ == "__main__":
    try:
        env = AmazingGameEnv()
        model_path = select_model_with_curses()
        print(f"\nLoading model: {model_path}")
        agent = load_agent(env, model_path)
        play_game(env, agent)
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        if 'env' in locals():
            env.close()