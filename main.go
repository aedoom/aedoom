// Copyright 2025 The aedoom Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"embed"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/pointlander/gradient/tf32"

	"github.com/AndreRenaud/gore"
	"github.com/hajimehoshi/ebiten/v2"
)

//go:embed doom1.wad
var DoomWad embed.FS

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

type (
	// Action is an action to take
	TypeAction uint
)

const (
	// ActionLeft
	ActionLeft TypeAction = iota
	// ActionRight
	ActionRight
	// ActionForward
	ActionForward
	// ActionBackward
	ActionBackward
	// ActionNone
	ActionNone
	// ActionActivate
	ActionActivate
	// ActionCount
	ActionCount
)

// Robot is a robot
type Robot interface {
	Init()
	Do(action TypeAction)
	Done()
}

// AutoEncoder is an autoencoder
type AutoEncoder struct {
	Set       tf32.Set
	Rng       *rand.Rand
	Iteration int
}

// Frame is a video frame
type Frame struct {
	Frame image.Image
}

// GrayAt returns the gray byte at
func (f *Frame) GrayAt(x, y int) color.Gray {
	return color.GrayModel.Convert(f.Frame.At(x, y)).(color.Gray)
}

// Order is the markov order
const (
	// Actions is the number of doom actions
	Actions = 6
	// Order is the order of the markov model
	Order = 2
)

// State is a markov state
type State [Order]TypeAction

// NewAutoEncoder creates a new autoencoder
func NewAutoEncoder(size int, markov bool) *AutoEncoder {
	a := AutoEncoder{
		Rng: rand.New(rand.NewSource(1)),
	}
	extra := 0
	if markov {
		extra = Order * Actions
	}
	set := tf32.NewSet()
	set.Add("l1", size+extra, size/2)
	set.Add("b1", size/2, 1)
	set.Add("l2", size, size+extra)
	set.Add("b2", size+extra, 1)

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, float32(a.Rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float32, len(w.X))
		}
	}

	a.Set = set
	return &a
}

// Measure measures the loss of a single input
func (a *AutoEncoder) Measure(input, output []float32, state *State) float32 {
	others := tf32.NewSet()
	size := len(input)
	if state != nil {
		size += Order * Actions
	}
	others.Add("input", size, 1)
	others.Add("output", size, 1)
	in := others.ByName["input"]
	for _, value := range input {
		in.X = append(in.X, value)
	}
	out := others.ByName["output"]
	for _, value := range output {
		out.X = append(out.X, value)
	}
	if state != nil {
		for _, v := range state {
			var s [Actions]float32
			s[v] = 1
			in.X = append(in.X, s[:]...)
			out.X = append(out.X, s[:]...)
		}
	}

	l1 := tf32.Everett(tf32.Add(tf32.Mul(a.Set.Get("l1"), others.Get("input")), a.Set.Get("b1")))
	l2 := tf32.Add(tf32.Mul(a.Set.Get("l2"), l1), a.Set.Get("b2"))
	loss := tf32.Sum(tf32.Quadratic(l2, others.Get("output")))

	l := float32(0.0)
	loss(func(a *tf32.V) bool {
		l = a.X[0]
		return true
	})
	return l
}

func (a *AutoEncoder) pow(x float64) float64 {
	y := math.Pow(x, float64(a.Iteration+1))
	if math.IsNaN(y) || math.IsInf(y, 0) {
		return 0
	}
	return y
}

// Encode encodes a single input
func (a *AutoEncoder) Encode(input, output []float32, rng *rand.Rand, state *State) float32 {
	others := tf32.NewSet()
	size := len(input)
	if state != nil {
		size += Order * Actions
	}
	others.Add("input", size, 1)
	others.Add("output", size, 1)
	in := others.ByName["input"]
	for _, value := range input {
		in.X = append(in.X, value)
	}
	out := others.ByName["output"]
	for _, value := range output {
		out.X = append(out.X, value)
	}
	if state != nil {
		for _, v := range state {
			var s [Actions]float32
			s[v] = 1
			in.X = append(in.X, s[:]...)
			out.X = append(out.X, s[:]...)
		}
	}

	dropout := map[string]interface{}{
		"rng": rng,
	}

	l1 := tf32.Dropout(tf32.Everett(tf32.Add(tf32.Mul(a.Set.Get("l1"), others.Get("input")), a.Set.Get("b1"))), dropout)
	l2 := tf32.Add(tf32.Mul(a.Set.Get("l2"), l1), a.Set.Get("b2"))
	loss := tf32.Avg(tf32.Quadratic(l2, others.Get("output")))

	l := float32(0.0)
	a.Set.Zero()
	others.Zero()
	l = tf32.Gradient(loss).X[0]
	if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
		fmt.Println(a.Iteration, l)
		return 0
	}

	norm := 0.0
	for _, p := range a.Set.Weights {
		for _, d := range p.D {
			norm += float64(d * d)
		}
	}
	norm = math.Sqrt(norm)
	b1, b2 := a.pow(B1), a.pow(B2)
	scaling := 1.0
	if norm > 1 {
		scaling = 1 / norm
	}
	for _, w := range a.Set.Weights {
		for ii, d := range w.D {
			g := d * float32(scaling)
			m := B1*w.States[StateM][ii] + (1-B1)*g
			v := B2*w.States[StateV][ii] + (1-B2)*g*g
			w.States[StateM][ii] = m
			w.States[StateV][ii] = v
			mhat := m / (1 - float32(b1))
			vhat := v / (1 - float32(b2))
			if vhat < 0 {
				vhat = 0
			}
			w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
	}
	a.Iteration++
	return l
}

func main() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		os.Exit(1)
	}()

	game := &DoomGame{}
	game.rng = rand.New(rand.NewSource(1))
	for i := range game.mind {
		game.mind[i].Auto = NewAutoEncoder(6, true)
		game.mind[i].Action = TypeAction(i)
	}
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Gamepad (Ebitengine Demo)")
	ebiten.SetFullscreen(false)
	go func() {
		gore.SetVirtualFileSystem(DoomWad)
		gore.Run(game, []string{"-iwad", "doom1.wad"})
		game.terminating = true
	}()
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
