package main

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/AndreRenaud/gore"
	"github.com/alixaxel/pagerank"
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
	counts    [Actions]float32
	iteration int
	state     State
	w, h      int
	autoMode  bool
	last      TypeAction
	input     []float32
	output    []float32
	index     int
	circular  [8]Matrix[float32]
}

var ActionMap = map[TypeAction]uint8{
	ActionLeft:     gore.KEY_LEFTARROW1,
	ActionRight:    gore.KEY_RIGHTARROW1,
	ActionForward:  gore.KEY_UPARROW1,
	ActionBackward: gore.KEY_DOWNARROW1,
	ActionNone:     0,
	ActionActivate: gore.KEY_USE1,
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
			event.Key = ActionMap[g.last]
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
	for y := 0; y < height-8; y += 8 {
		for x := 0; x < width-8; x += 8 {
			input, output := make([]float32, 8*8), make([]float32, 8*8)
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					pixel := float32(img.GrayAt(x+xx, y+yy).Y) / 255
					output[yy*8+xx] = pixel
					input[yy*8+xx] = pixel
				}
			}
			pixels = append(pixels, Patch{
				Input:  input,
				Output: output,
			})
		}
	}

	indexes := rand.Perm((g.w - 1) * (g.h - 1))
	indexes = indexes[:len(indexes)/Scale]

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
			Entropy: max,
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
		if act.Max >= 0 {
			g.votes[act.Max] += act.Entropy
			g.counts[act.Max]++
		}
		flight--

		go measure(indexes[index], g.rng.Int63())
		flight++
		index++
	}
	for range flight {
		act := <-done
		if act.Max >= 0 {
			g.votes[act.Max] += act.Entropy
			g.counts[act.Max]++
		}
	}
	if g.iteration%30 == 0 {
		for i := range g.votes {
			if g.counts[i] == 0 {
				continue
			}
			g.votes[i] /= g.counts[i]
		}
		copy(g.circular[g.index].Data, g.votes[:])
		g.index = (g.index + 1) % len(g.circular)
		graph := pagerank.NewGraph()
		for i := range g.circular {
			for ii := range g.circular {
				x, y := (i+g.index)%len(g.circular), (ii+g.index)%len(g.circular)
				cs := g.circular[x].CS(g.circular[y])
				graph.Link(uint32(x), uint32(y), float64(cs))
			}
		}
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			g.input[node] = float32(rank)
			g.output[node] = float32(rank)
		})
		max, action := float32(0.0), TypeAction(0)
		for i := range g.mind {
			value := g.mind[i].Auto.Measure(g.input, g.output, &g.state)
			if value > max {
				max, action = value, g.mind[i].Action
			}
		}
		g.mind[action].Auto.Encode(g.input, g.output, g.rng, &g.state)
		for i := range g.votes {
			g.votes[i] = 0
			g.counts[i] = 0
		}
		if g.autoMode && g.last != ActionCount {
			var event gore.DoomEvent
			event.Type = gore.Ev_keyup
			event.Key = ActionMap[g.last]
			g.events = append(g.events, event)
		}
		if g.autoMode {
			var event gore.DoomEvent
			event.Type = gore.Ev_keydown
			event.Key = ActionMap[action]
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
