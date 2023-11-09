import pygame



def main():
    pygame.init()
    WIDTH = 1000
    HEIGHT = 900
    font_type = "/usr/share/fonts/truetype/noto/NotoSansDisplay-ExtraCondensedItalic.ttf"

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    pygame.display.set_caption('Two-Player Pygame Chess!')

    font = pygame.font.Font(font_type, 20)
    big_font = pygame.font.Font(font_type, 50)

    timer = pygame.time.Clock()
    fps = 60

    # game variables and images

    run = True

    while run:
        timer.tick(fps)
        screen.fill('dark gray')

        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        pygame.display.flip()
    
    pygame.quit()

if __name__=="__main__":
    main()