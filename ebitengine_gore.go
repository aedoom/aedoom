package main

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/AndreRenaud/gore"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

const (
	screenWidth  = 640
	screenHeight = 480
)

type Auto struct {
	Auto   *AutoEncoder
	Action TypeAction
}

type DoomGame struct {
	lastFrame *ebiten.Image

	events      []gore.DoomEvent
	lock        sync.Mutex
	terminating bool

	rng       *rand.Rand
	auto      [][Actions]Auto
	mind      [Actions]Auto
	votes     [Actions]float32
	iteration int
	state     State
	w, h      int
	autoMode  bool
	last      TypeAction
}

func (g *DoomGame) Update() error {
	keys := map[ebiten.Key]uint8{
		ebiten.KeySpace:     gore.KEY_USE1,
		ebiten.KeyEscape:    gore.KEY_ESCAPE,
		ebiten.KeyUp:        gore.KEY_UPARROW1,
		ebiten.KeyDown:      gore.KEY_DOWNARROW1,
		ebiten.KeyLeft:      gore.KEY_LEFTARROW1,
		ebiten.KeyRight:     gore.KEY_RIGHTARROW1,
		ebiten.KeyEnter:     gore.KEY_ENTER,
		ebiten.KeyControl:   gore.KEY_FIRE1,
		ebiten.KeyShift:     0x80 + 0x36,
		ebiten.KeyBackspace: gore.KEY_BACKSPACE3,
		ebiten.KeyY:         'y',
		ebiten.KeyN:         'n',
		ebiten.KeyI:         'i',
		ebiten.KeyD:         'd',
		ebiten.KeyF:         'f',
		ebiten.KeyA:         'a',
		ebiten.KeyE:         'e',
		ebiten.KeyR:         'r',
		ebiten.KeyV:         'v',
		ebiten.KeyC:         'c',
		ebiten.KeyL:         'l',
		ebiten.KeyQ:         'q',
		ebiten.Key1:         '1',
		ebiten.Key2:         '2',
		ebiten.Key3:         '3',
		ebiten.Key4:         '4',
		ebiten.Key5:         '5',
		ebiten.Key6:         '6',
		ebiten.Key7:         '7',
		ebiten.Key8:         '8',
		ebiten.Key9:         '9',
		ebiten.Key0:         '0',
	}
	g.lock.Lock()
	defer g.lock.Unlock()
	if inpututil.IsKeyJustPressed(ebiten.KeyK) {
		g.autoMode = !g.autoMode
		if !g.autoMode && g.last != ActionCount {
			var event gore.DoomEvent
			event.Type = gore.Ev_keyup
			switch g.last {
			case ActionLeft:
				event.Key = gore.KEY_LEFTARROW1
			case ActionRight:
				event.Key = gore.KEY_RIGHTARROW1
			case ActionForward:
				event.Key = gore.KEY_UPARROW1
			case ActionBackward:
				event.Key = gore.KEY_DOWNARROW1
			case ActionNone:
			case ActionActivate:
				event.Key = gore.KEY_USE1
			}
			g.events = append(g.events, event)
			g.last = ActionCount
		}
	}
	for key, doomKey := range keys {
		if inpututil.IsKeyJustPressed(key) {
			var event gore.DoomEvent

			event.Type = gore.Ev_keydown
			event.Key = doomKey
			g.events = append(g.events, event)
		} else if inpututil.IsKeyJustReleased(key) {
			var event gore.DoomEvent
			event.Type = gore.Ev_keyup
			event.Key = doomKey
			g.events = append(g.events, event)
		}

		var mouseEvent gore.DoomEvent
		x, y := ebiten.CursorPosition()
		mouseEvent.Mouse.XPos = float64(x) / float64(screenWidth)
		mouseEvent.Mouse.YPos = float64(y) / float64(screenHeight)
		mouseEvent.Type = gore.Ev_mouse
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
			mouseEvent.Mouse.Button1 = true
		}
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight) {
			mouseEvent.Mouse.Button2 = true
		}
		g.events = append(g.events, mouseEvent)
	}
	if g.terminating {
		return ebiten.Termination
	}
	return nil
}

func (g *DoomGame) Draw(screen *ebiten.Image) {
	g.lock.Lock()
	defer g.lock.Unlock()

	if g.lastFrame == nil {
		return
	}
	op := &ebiten.DrawImageOptions{}
	rect := g.lastFrame.Bounds()
	yScale := float64(screenHeight) / float64(rect.Dy())
	xScale := float64(screenWidth) / float64(rect.Dx())
	op.GeoM.Scale(xScale, yScale)
	screen.DrawImage(g.lastFrame, op)
}

func (g *DoomGame) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func (g *DoomGame) GetEvent(event *gore.DoomEvent) bool {
	g.lock.Lock()
	defer g.lock.Unlock()
	if len(g.events) > 0 {
		*event = g.events[0]
		g.events = g.events[1:]
		return true
	}
	return false
}

func (g *DoomGame) DrawFrame(frame *image.RGBA) {
	g.lock.Lock()
	defer g.lock.Unlock()

	img := Frame{frame}
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if g.auto == nil {
		w, h := width/8, height/8
		fmt.Println(width, height, w, h, w*h)
		g.auto = make([][Actions]Auto, w*h)
		for i := range g.auto {
			for ii := range g.auto[i] {
				g.auto[i][ii].Auto = NewAutoEncoder(8*8, true)
				g.auto[i][ii].Action = TypeAction(ii)
			}
		}
		g.w, g.h = w, h
	}
	type Patch struct {
		Input   []float32
		Output  []float32
		Entropy float32
	}
	pixels := make([]Patch, 0, 8)
	rng := g.rng
	for y := 0; y < height-8; y += 8 {
		for x := 0; x < width-8; x += 8 {
			input, output := make([]float32, 8*8), make([]float32, 8*8)
			var histogram [256]float32
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					g := img.GrayAt(x+xx, y+yy)
					pixel := float32(g.Y) / 255
					output[yy*8+xx] = pixel
					pixel += float32(rng.NormFloat64() / 16)
					if pixel < 0 {
						pixel = 0
					}
					if pixel > 1 {
						pixel = 1
					}
					input[yy*8+xx] = pixel
					histogram[g.Y]++
				}
			}
			entropy := float32(0.0)
			for _, value := range histogram {
				if value == 0 {
					continue
				}
				entropy += (value / (float32(8 * 8))) * float32(math.Log2(float64(value)/float64(8*8)))
			}
			pixels = append(pixels, Patch{
				Input:   input,
				Output:  output,
				Entropy: -entropy,
			})
		}
	}

	indexes := rand.Perm((g.w - 1) * (g.h - 1))
	indexes = indexes[:len(indexes)/64]

	type Vote struct {
		Min     int
		Max     int
		Entropy float32
	}
	done := make(chan Vote, 8)
	measure := func(i int, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		min, max, minIndex, maxIndex := float32(math.MaxFloat32), float32(0), 0, 0
		for ii := range g.auto[i] {
			value := g.auto[i][ii].Auto.Measure(pixels[i].Input, pixels[i].Output, &g.state)
			if value < min {
				min, minIndex = value, ii
			}
			if value > max {
				max, maxIndex = value, ii
			}
		}
		g.auto[i][maxIndex].Auto.Encode(pixels[i].Input, pixels[i].Output, rng, &g.state)
		done <- Vote{
			Min:     minIndex,
			Max:     maxIndex,
			Entropy: pixels[i].Entropy,
		}
	}
	index, flight, cpus := 0, 0, runtime.NumCPU()
	for index < len(indexes) && flight < cpus {
		go measure(indexes[index], g.rng.Int63())
		flight++
		index++
	}
	for index < len(indexes) {
		act := <-done
		if act.Min >= 0 {
			g.votes[act.Min] += act.Entropy
		}
		if act.Max >= 0 {
			g.votes[act.Max] += act.Entropy
		}
		flight--

		go measure(indexes[index], g.rng.Int63())
		flight++
		index++
	}
	for range flight {
		act := <-done
		if act.Min >= 0 {
			g.votes[act.Min] += act.Entropy
		}
		if act.Max >= 0 {
			g.votes[act.Max] += act.Entropy
		}
	}
	if g.iteration%30 == 0 {
		input, output := make([]float32, Actions), make([]float32, Actions)
		sum := float32(0.0)
		for _, value := range g.votes {
			sum += value
		}
		for i, value := range g.votes {
			input[i] = value / sum
			output[i] = value / sum
			g.votes[i] = 0.0
		}
		max, min, learn, action := float32(0.0), float32(math.MaxFloat32), TypeAction(0), TypeAction(0)
		for i := range g.mind {
			value := g.mind[i].Auto.Measure(input, output, &g.state)
			if value > max {
				max, learn = value, g.mind[i].Action
			}
			if value < min {
				min, action = value, g.mind[i].Action
			}
		}
		g.mind[learn].Auto.Encode(input, output, g.rng, &g.state)

		if g.autoMode && g.last != ActionCount {
			var event gore.DoomEvent
			event.Type = gore.Ev_keyup
			switch g.last {
			case ActionLeft:
				event.Key = gore.KEY_LEFTARROW1
			case ActionRight:
				event.Key = gore.KEY_RIGHTARROW1
			case ActionForward:
				event.Key = gore.KEY_UPARROW1
			case ActionBackward:
				event.Key = gore.KEY_DOWNARROW1
			case ActionNone:
			case ActionActivate:
				event.Key = gore.KEY_USE1
			}
			g.events = append(g.events, event)
		}
		if g.autoMode {
			var event gore.DoomEvent
			event.Type = gore.Ev_keydown
			switch action {
			case ActionLeft:
				event.Key = gore.KEY_LEFTARROW1
			case ActionRight:
				event.Key = gore.KEY_RIGHTARROW1
			case ActionForward:
				event.Key = gore.KEY_UPARROW1
			case ActionBackward:
				event.Key = gore.KEY_DOWNARROW1
			case ActionNone:
			case ActionActivate:
				event.Key = gore.KEY_USE1
			}
			g.events = append(g.events, event)
		}
		g.last = action

		pre := TypeAction(action)
		for ii, value := range g.state {
			g.state[ii], pre = pre, value
		}
	}

	g.iteration++

	if g.lastFrame != nil {
		if g.lastFrame.Bounds().Dx() != frame.Bounds().Dx() || g.lastFrame.Bounds().Dy() != frame.Bounds().Dy() {
			g.lastFrame.Deallocate()
			g.lastFrame = nil
		}
	}
	if g.lastFrame == nil {
		g.lastFrame = ebiten.NewImage(frame.Bounds().Dx(), frame.Bounds().Dy())
	}
	g.lastFrame.WritePixels(frame.Pix)
}

func (g *DoomGame) SetTitle(title string) {
	ebiten.SetWindowTitle(title)
}
